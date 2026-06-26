"""
Convert MCAP human data files → zarr dataset.
Mirrors zarr_human_data_conversion.py but reads from .mcap instead of
episode.pkl + recording.svo2.

Key difference: quest2camera calibration is embedded in each .mcap file,
so --calib_quest2camera_file is optional (MCAP-embedded value takes priority).

Usage:
    python -m scripts_data.entry.zarr_mcap_data_conversion \
        -i data/mcap_sessions/human_se_task_name \
        -o data/zarr_data/zarr_data_human \
        -ins "task_name" \
        -m o
"""

import os
import pickle
import shutil
import pathlib
import numpy as np
import cv2
import click
from tqdm import tqdm
from multiprocessing import Process
from typing import Union, Dict, Optional
from scipy.spatial.transform import Rotation

from common.replay_buffer import ReplayBuffer
from common.cv2_util import get_image_transform_resize_crop, intrinsic_transform_resize
from common.timestamp_accumulator import get_accumulate_timestamp_idxs
from common.interpolation_util import PoseInterpolator, get_interp1d
from common.pose_util import euler_pose_to_mat, mat_to_pose, mat_to_euler_pose, pose_to_mat
from common.mcap_reader import MCAPReader
from human_data.constants import yfxrzu2standard
from human_data.hand_retargeting import Hand_Retargeting

hand_retargeting = Hand_Retargeting("./real/teleop/inspire_hand_0_4_6.yml")


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def conversion_single_mcap(
    mcap_path: str,
    calib_quest2camera_override: Optional[np.ndarray],
    speed_downsample_ratio: Optional[float],
    hand_shrink_coef: float,
    mode: str,
    out_resolutions_resize: tuple,
    out_resolutions_crop: tuple,
    out_resolutions_image_final: tuple,
    network_delay_checking: float = 1.0,
):
    mcap = MCAPReader(mcap_path)

    # --- calibration: MCAP-embedded takes priority, override is fallback ---
    quest2camera = mcap.quest2camera
    if quest2camera is None:
        if calib_quest2camera_override is None:
            print(f"[Warning] No quest2camera in {mcap_path} and no override provided, skip.")
            return None
        quest2camera = calib_quest2camera_override
        print(f"[Info] Using external quest2camera for {mcap_path}")

    # --- load Quest episode dict ---
    episode_org = mcap.load_episode()   # left_hand_mat, right_hand_mat, head_pose_mat, timestamp

    if len(episode_org["timestamp"]) < 10:
        print(f"[Warning] Too few frames in {mcap_path}, skip.")
        return None

    # --- network delay check ---
    dt_check = episode_org["timestamp"][1:] - episode_org["timestamp"][:-1]
    if dt_check.max() > network_delay_checking:
        print(f"[Warning] Max delay {dt_check.max():.3f}s > {network_delay_checking}, skip {mcap_path}")
        return None

    # --- resample Quest data to uniform dt ---
    start_time_idx = 0
    eps = 0.005
    for i in range(1, len(episode_org["right_hand_mat"])):
        if np.linalg.norm(episode_org["right_hand_mat"][i, 6:9] - episode_org["right_hand_mat"][0, 6:9]) > eps:
            start_time_idx = i
            break
    for key in episode_org:
        episode_org[key] = episode_org[key][start_time_idx:]

    t_init = 4
    for key in episode_org:
        episode_org[key] = episode_org[key][t_init:]

    downsample_ratio = speed_downsample_ratio if speed_downsample_ratio is not None else 1.0
    dt = (episode_org["timestamp"][1:] - episode_org["timestamp"][:-1]).mean() / downsample_ratio
    T_start = episode_org["timestamp"][0]
    T_end = episode_org["timestamp"][-1]
    n_steps = int((T_end - T_start) / dt)
    timestamps = np.arange(n_steps + 1) * dt + T_start

    len_hand_arr = len(episode_org["left_hand_mat"][0])
    actions = np.concatenate(
        [episode_org["left_hand_mat"], episode_org["right_hand_mat"], episode_org["head_pose_mat"]], axis=1
    )
    n_pose = actions.shape[1] // 6
    actions_record = []
    for i in range(n_pose):
        act_rotvec = mat_to_pose(euler_pose_to_mat(actions[:, i * 6:(i + 1) * 6]))
        interp = PoseInterpolator(episode_org["timestamp"], act_rotvec)
        act_euler = mat_to_euler_pose(pose_to_mat(interp(timestamps)))
        actions_record.append(act_euler)
    actions_record = np.concatenate(actions_record, axis=1)

    episode_org["left_hand_mat"] = actions_record[:, :len_hand_arr]
    episode_org["right_hand_mat"] = actions_record[:, len_hand_arr:2 * len_hand_arr]
    episode_org["head_pose_mat"] = actions_record[:, 2 * len_hand_arr:]
    episode_org["head_pose_mat"] = mat_to_euler_pose(euler_pose_to_mat(episode_org["head_pose_mat"]))

    # --- transform to egocentric (T0 camera) frame ---
    episode_org["head_pose_mat"] = euler_pose_to_mat(episode_org["head_pose_mat"]) @ yfxrzu2standard
    vr2camera0 = quest2camera @ fast_mat_inv(episode_org["head_pose_mat"][0])
    camera_pose = vr2camera0 @ episode_org["head_pose_mat"] @ fast_mat_inv(quest2camera)

    b_hand_joint = len_hand_arr // 6
    left_hand_pose, right_hand_pose = [], []
    left_hand_pose_rotvec, right_hand_pose_rotvec = [], []
    for i in range(b_hand_joint):
        left_pose = vr2camera0 @ (euler_pose_to_mat(episode_org["left_hand_mat"][:, i * 6:(i + 1) * 6]) @ yfxrzu2standard)
        right_pose = vr2camera0 @ (euler_pose_to_mat(episode_org["right_hand_mat"][:, i * 6:(i + 1) * 6]) @ yfxrzu2standard)
        left_hand_pose.append(mat_to_euler_pose(left_pose))
        right_hand_pose.append(mat_to_euler_pose(right_pose))
        left_hand_pose_rotvec.append(mat_to_pose(left_pose))
        right_hand_pose_rotvec.append(mat_to_pose(right_pose))

    left_hand_pose = np.concatenate(left_hand_pose, axis=1)
    right_hand_pose = np.concatenate(right_hand_pose, axis=1)
    left_hand_pose_rotvec = np.concatenate(left_hand_pose_rotvec, axis=-1)
    right_hand_pose_rotvec = np.concatenate(right_hand_pose_rotvec, axis=-1)

    # --- hand retargeting ---
    (left_hand_wrists, right_hand_wrists, left_hand_fix_wrists, right_hand_fix_wrists,
     left_hand_qposes, right_hand_qposes, left_hand_urdf_qposes, right_hand_urdf_qposes,
     left_org_hand_poses, right_org_hand_poses) = hand_retargeting.retarget(left_hand_pose, right_hand_pose)

    left_hand_wrists = mat_to_pose(euler_pose_to_mat(left_hand_wrists))
    right_hand_wrists = mat_to_pose(euler_pose_to_mat(right_hand_wrists))
    left_hand_fix_wrists = mat_to_pose(euler_pose_to_mat(left_hand_fix_wrists))
    right_hand_fix_wrists = mat_to_pose(euler_pose_to_mat(right_hand_fix_wrists))
    left_org_hand_poses = mat_to_pose(euler_pose_to_mat(left_org_hand_poses.reshape(-1, 6))).reshape(-1, 5, 6)
    right_org_hand_poses = mat_to_pose(euler_pose_to_mat(right_org_hand_poses.reshape(-1, 6))).reshape(-1, 5, 6)

    episode = {
        "timestamp": timestamps,
        "left_hand_pose": left_hand_pose_rotvec,
        "right_hand_pose": right_hand_pose_rotvec,
        "left_wrist_pose": left_hand_wrists,
        "right_wrist_pose": right_hand_wrists,
        "left_wrist_fix_pose": left_hand_fix_wrists,
        "right_wrist_fix_pose": right_hand_fix_wrists,
        "left_finger_pose": left_org_hand_poses,
        "right_finger_pose": right_org_hand_poses,
        "camera0_pose": mat_to_pose(camera_pose),
        "robot0_eef_pos": right_hand_fix_wrists[:, :3],
        "robot0_eef_rot_axis_angle": right_hand_fix_wrists[:, 3:],
        "robot1_eef_pos": left_hand_fix_wrists[:, :3],
        "robot1_eef_rot_axis_angle": left_hand_fix_wrists[:, 3:],
    }

    if hand_shrink_coef is not None and hand_shrink_coef != 1.0:
        right_delta = (right_hand_qposes[1:] - right_hand_qposes[:-1]) * hand_shrink_coef
        left_delta = (left_hand_qposes[1:] - left_hand_qposes[:-1]) * hand_shrink_coef
        right_hand_qposes = np.concatenate([right_hand_qposes[:1], right_hand_qposes[:1] + np.cumsum(right_delta, axis=0)], axis=0)
        left_hand_qposes = np.concatenate([left_hand_qposes[:1], left_hand_qposes[:1] + np.cumsum(left_delta, axis=0)], axis=0)

    episode["gripper0_gripper_pose"] = right_hand_qposes
    episode["gripper1_gripper_pose"] = left_hand_qposes

    # --- align and extract camera frames ---
    episode_length = len(timestamps)
    camera_timestamps_sec = mcap.get_camera_timestamps()   # (N_cam,) Unix seconds
    camera_timestamps_ms = (camera_timestamps_sec * 1e3).astype(np.int64)
    quest_timestamps_ms = (timestamps * 1e3).astype(np.int64)

    cam_idxs, quest_idxs = get_accumulate_timestamp_idxs(
        camera_timestamps_ms, quest_timestamps_ms
    )

    camera_info = mcap.get_camera_information()
    intrinsics = intrinsic_transform_resize(
        camera_info["left_intrinsic"],
        input_res=mcap.get_frame_resolution(),
        output_resize_res=out_resolutions_resize,
        output_crop_res=out_resolutions_crop,
    )
    intrinsics_final = intrinsic_transform_resize(
        intrinsics,
        input_res=out_resolutions_crop,
        output_resize_res=out_resolutions_image_final,
        output_crop_res=out_resolutions_image_final,
    )
    episode["camera0_left_intrinsic"] = np.tile(intrinsics, (episode_length, 1, 1))
    episode["camera0_left_intrinsic_final"] = np.tile(intrinsics_final, (episode_length, 1, 1))

    transform_img = get_image_transform_resize_crop(
        input_res=mcap.get_frame_resolution(),
        output_resize_res=out_resolutions_resize,
        output_crop_res=out_resolutions_crop,
        bgr_to_rgb=False,
    )
    transform_final = get_image_transform_resize_crop(
        input_res=out_resolutions_crop,
        output_resize_res=out_resolutions_image_final,
        output_crop_res=out_resolutions_image_final,
        bgr_to_rgb=False,
    )

    rgb_list = [None] * episode_length
    for cam_i, quest_i in zip(cam_idxs, quest_idxs):
        if quest_i >= episode_length:
            break
        mcap.set_frame_index(cam_i)
        result = mcap.read_camera(return_timestamp=False)
        if result is None:
            continue
        frame_key = list(result["image"].keys())[0]
        rgb = result["image"][frame_key]
        rgb = transform_img(rgb)
        rgb = transform_final(rgb)
        rgb_list[quest_i] = rgb

    # fill gaps (nearest neighbor)
    last_valid = None
    for i, rgb in enumerate(rgb_list):
        if rgb is not None:
            last_valid = rgb
        elif last_valid is not None:
            rgb_list[i] = last_valid
    rgb_arr = np.stack([r for r in rgb_list if r is not None])
    if len(rgb_arr) != episode_length:
        print(f"[Warning] Camera frame count mismatch: {len(rgb_arr)} vs {episode_length}, skip.")
        return None
    episode["camera0_rgb"] = rgb_arr

    return episode


def conversion_trajectory_mcap(
    mcap_paths, calib_override, speed_downsample_ratio, hand_shrink_coef,
    mode, out_resolutions_resize, out_resolutions_crop, resolution_image_final,
    network_delay_checking, replay_buffer, process_id,
):
    pbar = tqdm(mcap_paths, desc=f"Process {process_id}")
    for mcap_path, source, source_idx in pbar:
        episode = conversion_single_mcap(
            mcap_path, calib_override, speed_downsample_ratio, hand_shrink_coef,
            mode, out_resolutions_resize, out_resolutions_crop, resolution_image_final,
            network_delay_checking,
        )
        if episode is None:
            pbar.write(f"Skip {mcap_path}")
            continue
        episode["embodiment"] = np.zeros((len(episode["robot0_eef_pos"]), 1))
        episode["source_idx"] = np.ones((len(episode["robot0_eef_pos"]), 1)) * source_idx
        replay_buffer.add_episode(episode, compressors="disk")


@click.command()
@click.option("--input_dir", "-i", required=True)
@click.option("--output", "-o", required=True)
@click.option("--instruction", "-ins", type=str, required=True)
@click.option("--calib_quest2camera_file", "-cf", default=None,
              help="Fallback calib .npy if not embedded in MCAP.")
@click.option("--speed_downsample_ratio", "-spr", default=1.0, type=float)
@click.option("--hand_shrink_coef", "-hsc", default=1.25, type=float)
@click.option("--mode", "-m", default="o",
              type=click.Choice(["o", "p", "s", "a"], case_sensitive=False))
@click.option("--resolution_resize", "-ror", default="640x480")
@click.option("--resolution_crop", "-or", default="640x480")
@click.option("--resolution_image_final", "-for", default="224x224")
@click.option("--network_delay_checking", "-dl", default=0.5, type=float)
@click.option("--n_encoding_threads", "-ne", default=-1, type=int)
def main(input_dir, output, instruction, calib_quest2camera_file,
         speed_downsample_ratio, hand_shrink_coef, mode,
         resolution_resize, resolution_crop, resolution_image_final,
         network_delay_checking, n_encoding_threads):

    out_resize = tuple(int(x) for x in resolution_resize.split("x"))
    out_crop = tuple(int(x) for x in resolution_crop.split("x"))
    out_final = tuple(int(x) for x in resolution_image_final.split("x"))

    calib_override = np.load(calib_quest2camera_file) if calib_quest2camera_file else None

    instruction = instruction.strip(".").replace(" ", "_")
    input_folder = os.path.basename(input_dir.rstrip("/"))
    replay_buffer_fp = pathlib.Path(
        os.path.join(output, f"{input_folder}_{resolution_resize}_{resolution_crop}+{instruction}+.zarr")
    )
    if replay_buffer_fp.exists():
        click.confirm("Output path already exists! Overwrite?", abort=True)
        shutil.rmtree(replay_buffer_fp)

    replay_buffer = ReplayBuffer.create_from_path(replay_buffer_fp, mode="a")

    # collect all .mcap files
    mcap_files = sorted(
        [(os.path.join(input_dir, f), "default", 0)
         for f in os.listdir(input_dir) if f.endswith(".mcap")]
    )
    print(f"Found {len(mcap_files)} MCAP files in {input_dir}")

    if n_encoding_threads > 1:
        batches = [mcap_files[i::n_encoding_threads] for i in range(n_encoding_threads)]
        procs = []
        for pid, batch in enumerate(batches):
            p = Process(target=conversion_trajectory_mcap, args=(
                batch, calib_override, speed_downsample_ratio, hand_shrink_coef,
                mode, out_resize, out_crop, out_final,
                network_delay_checking, replay_buffer, pid,
            ))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        conversion_trajectory_mcap(
            mcap_files, calib_override, speed_downsample_ratio, hand_shrink_coef,
            mode, out_resize, out_crop, out_final,
            network_delay_checking, replay_buffer, 0,
        )

    print("Done.")


if __name__ == "__main__":
    main()
