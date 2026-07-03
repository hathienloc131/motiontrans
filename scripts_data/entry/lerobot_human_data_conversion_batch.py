"""
LeRobot human VR data conversion — mirrors zarr_human_data_conversion_batch.py but
writes to LeRobot format (parquet + MP4) instead of zarr.

We intentionally do NOT import zarr_human_data_conversion_batch to avoid pulling in
ReplayBuffer → zarr → numcodecs, none of which are needed here.
conversion_single_trajectory and its helpers are copied directly.
"""
import sys
import os
import json
import pathlib
import shutil
import pickle
from pathlib import Path
from typing import Dict, Union

import click
import cv2
import imageio
import numpy as np
import fpsample
from tqdm import tqdm

try:
    from common.svo_utils import SVOReader
except Exception:
    SVOReader = None

from common.cv2_util import get_image_transform_resize_crop, intrinsic_transform_resize
from common.timestamp_accumulator import get_accumulate_timestamp_idxs
from scipy.spatial.transform import Rotation as _Rotation
from common.pose_util import euler_pose_to_mat, mat_to_pose, mat_to_euler_pose, pose_to_mat
from common.interpolation_util import PoseInterpolator, get_interp1d
from human_data.constants import yfxrzu2standard
from human_data.hand_retargeting import Hand_Retargeting

hand_retargeting = Hand_Retargeting("./real/teleop/inspire_hand_0_4_6.yml")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (copied from zarr_human_data_conversion_batch to avoid zarr import)
# ─────────────────────────────────────────────────────────────────────────────

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def _peek_vr_hand_dim(input_data_fp_list: list) -> int:
    """Read the first available episode.pkl to get the raw VR hand joint dimension."""
    for save_dir, _, _ in input_data_fp_list:
        ep_path = os.path.join(save_dir, "episode.pkl")
        if os.path.exists(ep_path):
            with open(ep_path, "rb") as f:
                ep = pickle.load(f)
            return len(ep["left_hand_mat"][0])
    return 0


def conversion_single_trajectory(
    mode,
    save_dir,
    calib_quest2camera,
    speed_downsample_ratio,
    single_arm,
    hand_shrink_coef,
    gripper_type: str = 'inspire_hand',  # 'inspire_hand' | 'gripper_f285'
    f285_close_ramp_k: int = 5,
    save_raw_vr: bool = False,
    out_resolutions_resize: Union[None, tuple, Dict[str, tuple]] = None,
    out_resolutions_crop: Union[None, tuple, Dict[str, tuple]] = None,
    out_resolutions_image_final: Union[None, tuple, Dict[str, tuple]] = None,
    network_delay_checking: float = 1.0,
    num_points_final: int = 2048,
    points_max_distance_final: float = 1.25,
):
    episode_path = os.path.join(save_dir, "episode.pkl")
    if not os.path.exists(episode_path):
        print(f"[Warning] No episode.pkl found in {save_dir}")
        return None
    with open(episode_path, "rb") as f:
        episode_org = pickle.load(f)

    start_time_idx = 0
    eps = 0.005
    for i in range(1, len(episode_org["right_hand_mat"])):
        if np.linalg.norm(episode_org["right_hand_mat"][i, 6:9] - episode_org["right_hand_mat"][0, 6:9]) > eps:
            start_time_idx = i
            break
    for key in episode_org.keys():
        episode_org[key] = episode_org[key][start_time_idx:]

    downsample_ratio = speed_downsample_ratio if speed_downsample_ratio is not None else 1.0
    dt = (episode_org["timestamp"][1:] - episode_org["timestamp"][:-1]).mean() / downsample_ratio

    print(f"speed_downsample_ratio: {downsample_ratio}")
    print("!" * 25)

    t_init = 4
    for key in episode_org.keys():
        episode_org[key] = episode_org[key][t_init:]

    dt_check_max = np.max(episode_org["timestamp"][1:] - episode_org["timestamp"][:-1])
    if dt_check_max > network_delay_checking:
        print(f"[Warning] Max delay {dt_check_max} > {network_delay_checking}, abandon this episode. {save_dir}")
        return None

    T_start = episode_org["timestamp"][0]
    T_end = episode_org["timestamp"][-1]
    n_steps = int((T_end - T_start) / dt)
    timestamps = np.arange(n_steps + 1) * dt + T_start

    len_hand_arr = len(episode_org["left_hand_mat"][0])
    actions = np.concatenate([episode_org["left_hand_mat"], episode_org["right_hand_mat"], episode_org["head_pose_mat"]], axis=1)
    n_pose = actions.shape[1] // 6
    actions_record = []
    for i in range(n_pose):
        act_rotvec = mat_to_pose(euler_pose_to_mat(actions[:, i * 6:(i + 1) * 6]))
        interp = PoseInterpolator(t=episode_org["timestamp"], x=act_rotvec)
        act_euler = mat_to_euler_pose(pose_to_mat(interp(timestamps)))
        actions_record.append(act_euler)
    actions_record = np.concatenate(actions_record, axis=1)
    episode_org["left_hand_mat"] = actions_record[:, :len_hand_arr]
    episode_org["right_hand_mat"] = actions_record[:, len_hand_arr:2 * len_hand_arr]
    episode_org["head_pose_mat"] = actions_record[:, 2 * len_hand_arr:]

    if save_raw_vr:
        _raw_vr_left_hand  = episode_org["left_hand_mat"].astype(np.float32)   # (T, N*6) euler VR world
        _raw_vr_right_hand = episode_org["right_hand_mat"].astype(np.float32)  # (T, N*6) euler VR world
        _raw_vr_head_pose  = episode_org["head_pose_mat"].astype(np.float32)   # (T, 6)   euler VR world

    episode_org["head_pose_mat"] = euler_pose_to_mat(episode_org["head_pose_mat"]) @ yfxrzu2standard
    vr2camera0 = calib_quest2camera @ fast_mat_inv(episode_org["head_pose_mat"][0])
    camera_pose = vr2camera0 @ episode_org["head_pose_mat"] @ fast_mat_inv(calib_quest2camera)

    b_hand_joint = len(episode_org["left_hand_mat"][0]) // 6
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

    (left_hand_wrists, right_hand_wrists, left_hand_fix_wrists, right_hand_fix_wrists,
     left_hand_qposes, right_hand_qposes, left_hand_urdf_qposes, right_hand_urdf_qposes,
     left_org_hand_poses, right_org_hand_poses,
     left_f285_qposes, right_f285_qposes) = hand_retargeting.retarget(
         left_hand_pose, right_hand_pose, f285_close_ramp_k=f285_close_ramp_k)

    left_hand_wrists = mat_to_pose(euler_pose_to_mat(left_hand_wrists))
    right_hand_wrists = mat_to_pose(euler_pose_to_mat(right_hand_wrists))
    left_hand_fix_wrists = mat_to_pose(euler_pose_to_mat(left_hand_fix_wrists))
    right_hand_fix_wrists = mat_to_pose(euler_pose_to_mat(right_hand_fix_wrists))
    left_org_hand_poses = mat_to_pose(euler_pose_to_mat(left_org_hand_poses.reshape(-1, 6))).reshape(-1, 5, 6)
    right_org_hand_poses = mat_to_pose(euler_pose_to_mat(right_org_hand_poses.reshape(-1, 6))).reshape(-1, 5, 6)

    # ── cm → m ────────────────────────────────────────────────────────────────
    _S = 0.01
    right_hand_wrists[:, :3] *= _S;        left_hand_wrists[:, :3] *= _S
    right_hand_fix_wrists[:, :3] *= _S;    left_hand_fix_wrists[:, :3] *= _S
    right_org_hand_poses[:, :, :3] *= _S;  left_org_hand_poses[:, :, :3] *= _S
    for _j in range(right_hand_pose_rotvec.shape[1] // 6):
        right_hand_pose_rotvec[:, _j*6:_j*6+3] *= _S
        left_hand_pose_rotvec[:, _j*6:_j*6+3] *= _S
    camera_pose[:, :3, 3] *= _S  # scale before mat_to_pose
    # ─────────────────────────────────────────────────────────────────────────

    # For F285: shift EEF from wrist to pinch point (0.145 m along wrist Z / finger direction).
    if gripper_type == 'gripper_f285':
        _F285_PINCH_M = 0.145
        right_hand_fix_wrists[:, :3] += _Rotation.from_rotvec(right_hand_fix_wrists[:, 3:]).as_matrix()[:, :, 2] * _F285_PINCH_M
        if not single_arm:
            left_hand_fix_wrists[:, :3]  += _Rotation.from_rotvec(left_hand_fix_wrists[:, 3:]).as_matrix()[:, :, 2]  * _F285_PINCH_M

    episode = dict()
    episode["timestamp"] = timestamps
    episode_length = len(timestamps)
    episode["left_hand_pose"] = left_hand_pose_rotvec
    episode["right_hand_pose"] = right_hand_pose_rotvec
    episode["left_wrist_pose"] = left_hand_wrists
    episode["right_wrist_pose"] = right_hand_wrists
    episode["left_wrist_fix_pose"] = left_hand_fix_wrists
    episode["right_wrist_fix_pose"] = right_hand_fix_wrists
    episode["left_finger_pose"] = left_org_hand_poses
    episode["right_finger_pose"] = right_org_hand_poses
    episode["camera0_pose"] = mat_to_pose(camera_pose)
    episode["robot0_eef_pos"] = right_hand_fix_wrists[:, :3]
    episode["robot0_eef_rot_axis_angle"] = right_hand_fix_wrists[:, 3:]
    if not single_arm:
        episode["robot1_eef_pos"] = left_hand_fix_wrists[:, :3]
        episode["robot1_eef_rot_axis_angle"] = left_hand_fix_wrists[:, 3:]

    if gripper_type == 'inspire_hand':
        if hand_shrink_coef is not None and hand_shrink_coef != 1.0:
            def _shrink(q):
                delta = (q[1:] - q[:-1]) * hand_shrink_coef
                return np.concatenate([q[0:1], delta]).cumsum(axis=0)
            right_hand_qposes = _shrink(right_hand_qposes)
            left_hand_qposes = _shrink(left_hand_qposes)
            right_hand_urdf_qposes = _shrink(right_hand_urdf_qposes)
            left_hand_urdf_qposes = _shrink(left_hand_urdf_qposes)
        episode["gripper0_gripper_pose"] = right_hand_qposes
        episode["urdf_gripper0_gripper_pose"] = right_hand_urdf_qposes
        if not single_arm:
            episode["gripper1_gripper_pose"] = left_hand_qposes
            episode["urdf_gripper1_gripper_pose"] = left_hand_urdf_qposes
    elif gripper_type == 'gripper_f285':
        episode["gripper0_gripper_pose"] = right_f285_qposes[:, None]  # (T, 1)
        if not single_arm:
            episode["gripper1_gripper_pose"] = left_f285_qposes[:, None]  # (T, 1)
    else:
        raise ValueError(f"Unknown gripper_type: {gripper_type!r}. Choose 'inspire_hand' or 'gripper_f285'.")

    if save_raw_vr:
        episode["raw_vr_left_hand"]  = _raw_vr_left_hand   # (T, N*6) euler VR world
        episode["raw_vr_right_hand"] = _raw_vr_right_hand  # (T, N*6) euler VR world
        episode["raw_vr_head_pose"]  = _raw_vr_head_pose   # (T, 6)   euler VR world

    # ── Frame cut ──────────────────────────────────────────────────────────────
    frame_cut_fp = os.path.join(save_dir, "frame_cut.txt")
    frame_cut = None
    if os.path.exists(frame_cut_fp):
        with open(frame_cut_fp) as f:
            fc = f.read().strip()
        if fc.isdigit():
            frame_cut = int((episode_org["timestamp"][int(fc) - start_time_idx] - T_start) / dt)
        else:
            print(f"Frame cut {fc} is not a digit, set to None.")
    if frame_cut is not None:
        episode_length = min(episode_length, frame_cut)
    for k in episode:
        episode[k] = episode[k][:episode_length]

    episode["camera0_real_timestamp"] = np.zeros((episode_length,), dtype=np.float64)

    # ── Video source ───────────────────────────────────────────────────────────
    svo_path = os.path.join(save_dir, "recording.svo2")
    mp4_path = os.path.join(save_dir, "rgb.mp4")

    if os.path.exists(svo_path):
        if SVOReader is None:
            print("[Warning] ZED SDK (SVOReader) not available; skipping SVO video.")
            return None

        svo_stereo = mode in ["s", "a"]
        svo_depth = mode in ["d", "a"]
        svo_pointcloud = mode in ["p", "a"]

        with open(os.path.join(save_dir, "device_id.txt")) as f:
            serial_id = f.read().strip()
        svo_camera = SVOReader(svo_path, serial_number=serial_id)
        svo_camera.set_reading_parameters(image=True, depth=svo_depth, pointcloud=svo_pointcloud, concatenate_images=False)
        frame_count = svo_camera.get_frame_count()
        width, height = svo_camera.get_frame_resolution()

        camera_info = svo_camera.get_camera_information()
        camera_info["left_intrinsic"] = intrinsic_transform_resize(camera_info["left_intrinsic"], input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop)
        camera_info["right_intrinsic"] = intrinsic_transform_resize(camera_info["right_intrinsic"], input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop)
        camera_info["left_intrinsic_final"] = intrinsic_transform_resize(camera_info["left_intrinsic"], input_res=out_resolutions_crop, output_resize_res=out_resolutions_image_final, output_crop_res=out_resolutions_image_final)
        camera_info["right_intrinsic_final"] = intrinsic_transform_resize(camera_info["right_intrinsic"], input_res=out_resolutions_crop, output_resize_res=out_resolutions_image_final, output_crop_res=out_resolutions_image_final)
        for key in ["stereo_transform", "left_intrinsic", "right_intrinsic", "left_intrinsic_final", "right_intrinsic_final"]:
            episode["camera0_" + key] = np.array([camera_info[key]] * episode_length)

        next_global_idx = 0
        start_time = episode["timestamp"][0]
        transform_img = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop, bgr_to_rgb=True)
        obs_dict = {"rgb": ("image", f"{serial_id}_left", transform_img)}
        if svo_stereo:
            obs_dict["rgb_right"] = ("image", f"{serial_id}_right", transform_img)
        if svo_depth:
            td = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop, is_depth=True)
            obs_dict["depth"] = ("depth", f"{serial_id}_left", td)
        if svo_pointcloud:
            tp = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop, is_depth=True)
            obs_dict["pointcloud"] = ("pointcloud", f"{serial_id}_left", tp)

        global_idx = 0
        for _ in range(frame_count):
            svo_output = svo_camera.read_camera(return_timestamp=True)
            if svo_output is None:
                break
            data_dict, timestamp = svo_output
            timestamp = timestamp / 1000.0
            if timestamp < episode["timestamp"][0] - dt:
                continue

            local_idxs, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(
                timestamps=[timestamp], start_time=start_time, dt=dt, next_global_idx=next_global_idx)

            if len(global_idxs) > 0:
                for global_idx in global_idxs:
                    if global_idx == episode_length:
                        break
                    for key in obs_dict:
                        value = data_dict[obs_dict[key][0]][obs_dict[key][1]]
                        transform = obs_dict[key][2]
                        if value.shape[-1] == 4:
                            value = value[..., :3]
                        value = transform(value)
                        if "rgb" in key:
                            value = cv2.resize(value, out_resolutions_image_final, interpolation=cv2.INTER_LINEAR)
                        if "pointcloud" in key:
                            points_xyz = value.reshape(-1, 3)
                            points_rgb = obs_dict["rgb"][2](data_dict["image"][obs_dict[key][1]][..., :3]).reshape(-1, 3)
                            points = np.concatenate([points_xyz, points_rgb / 255.0], axis=-1)
                            valid_mask = np.linalg.norm(points_xyz, axis=-1) <= points_max_distance_final
                            points = points[valid_mask]
                            points_xyz = points_xyz[valid_mask]
                            if len(points) > num_points_final:
                                pts_idx = fpsample.bucket_fps_kdline_sampling(points_xyz, num_points_final, h=7)
                            else:
                                pts_idx = np.array([i % len(points_xyz) for i in range(num_points_final)])
                            value = points[pts_idx]
                        if "camera0_" + key not in episode:
                            episode["camera0_" + key] = np.zeros((episode_length,) + value.shape, dtype=value.dtype)
                        episode["camera0_" + key][global_idx] = value
                    episode["camera0_real_timestamp"][global_idx] = timestamp
            if (next_global_idx == episode_length) or (global_idx == episode_length):
                break

        if (next_global_idx < episode_length) and (global_idx != episode_length):
            abandoned = episode_length - next_global_idx
            for k in episode:
                try:
                    episode[k] = episode[k][:-abandoned]
                except Exception:
                    pass
            print(f"Warning: abandoned {abandoned} frames.")

    elif os.path.exists(mp4_path):
        if mode != "o":
            print(f"[Warning] RealSense MP4 only supports mode='o'; ignoring depth/stereo/pointcloud.")

        intrinsic_path = os.path.join(save_dir, "camera_intrinsic.npy")
        raw_intrinsic = np.load(intrinsic_path) if os.path.exists(intrinsic_path) else np.eye(3)

        reader = imageio.get_reader(mp4_path)
        meta = reader.get_meta_data()
        width, height = meta["size"]

        left_intrinsic = intrinsic_transform_resize(raw_intrinsic, input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop)
        left_intrinsic_final = intrinsic_transform_resize(left_intrinsic, input_res=out_resolutions_crop, output_resize_res=out_resolutions_image_final, output_crop_res=out_resolutions_image_final)
        for key, val in [("stereo_transform", np.eye(4)), ("left_intrinsic", left_intrinsic), ("right_intrinsic", left_intrinsic), ("left_intrinsic_final", left_intrinsic_final), ("right_intrinsic_final", left_intrinsic_final)]:
            episode["camera0_" + key] = np.array([val] * episode_length)

        ts_path = os.path.join(save_dir, "camera_timestamps.npy")
        if os.path.exists(ts_path):
            frame_timestamps = np.load(ts_path)
            n_frames = len(frame_timestamps)
        else:
            fps_vid = meta.get("fps", 30.0)
            cap = cv2.VideoCapture(mp4_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_timestamps = episode["timestamp"][0] + np.arange(n_frames) / fps_vid

        desired_frame_idxs = np.array([
            min(int(np.argmin(np.abs(frame_timestamps - episode["timestamp"][i]))), n_frames - 1)
            for i in range(episode_length)
        ])
        transform_img = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop, bgr_to_rgb=False)
        episode["camera0_rgb"] = np.zeros((episode_length, out_resolutions_image_final[1], out_resolutions_image_final[0], 3), dtype=np.uint8)

        needed_idxs = set(desired_frame_idxs.tolist())
        max_needed_idx = int(desired_frame_idxs.max()) if len(desired_frame_idxs) > 0 else 0
        frame_cache = {}
        for i, frame in enumerate(reader):
            if i in needed_idxs:
                frame_cache[i] = frame
            if i >= max_needed_idx:
                break
        reader.close()

        for global_idx in range(episode_length):
            frame_idx = desired_frame_idxs[global_idx]
            frame = transform_img(frame_cache[frame_idx])
            frame = cv2.resize(frame, out_resolutions_image_final, interpolation=cv2.INTER_LINEAR)
            episode["camera0_rgb"][global_idx] = frame
            episode["camera0_real_timestamp"][global_idx] = frame_timestamps[frame_idx]

    else:
        print(f"[Warning] No recording.svo2 or rgb.mp4 found in {save_dir}, skipping.")
        return None

    n_length = min(episode["timestamp"].shape[0], episode["camera0_real_timestamp"].shape[0])
    for k in episode:
        episode[k] = episode[k][:n_length]
    print(f"length: {n_length}")

    # ── Grasp annotation ───────────────────────────────────────────────────────
    def _read_frame_txt(fp):
        if not os.path.exists(fp):
            return None
        with open(fp) as f:
            v = f.read().strip()
        if v.isdigit():
            return int((episode_org["timestamp"][int(v) - start_time_idx] - T_start) / dt)
        print(f"Frame marker {v} is not a digit, ignoring.")
        return None

    frame_grasp = _read_frame_txt(os.path.join(save_dir, "frame_grasp.txt"))
    frame_release = _read_frame_txt(os.path.join(save_dir, "frame_release.txt"))
    print(f"frame_grasp: {frame_grasp}, frame_release: {frame_release}")

    if frame_grasp is not None:
        assert hand_shrink_coef == 1.0, "Cannot use hand_shrink_coef != 1.0 with frame_grasp/release handler."
        if frame_release is None:
            frame_release = n_length
        g0_max = np.max(episode["gripper0_gripper_pose"][frame_grasp:frame_release + 1], axis=0)
        assert frame_grasp >= 11
        episode["gripper0_gripper_pose"][frame_grasp:frame_release + 1] = g0_max[None]
        interp = get_interp1d(episode["timestamp"][[frame_grasp - 11, frame_grasp]], episode["gripper0_gripper_pose"][[frame_grasp - 11, frame_grasp]])
        episode["gripper0_gripper_pose"][frame_grasp - 10:frame_grasp] = interp(episode["timestamp"][frame_grasp - 10:frame_grasp])
        if not single_arm:
            g1_max = np.max(episode["gripper1_gripper_pose"][frame_grasp:frame_release + 1], axis=0)
            episode["gripper1_gripper_pose"][frame_grasp:frame_release + 1] = g1_max[None]
            interp1 = get_interp1d(episode["timestamp"][[frame_grasp - 11, frame_grasp]], episode["gripper1_gripper_pose"][[frame_grasp - 11, frame_grasp]])
            episode["gripper1_gripper_pose"][frame_grasp - 10:frame_grasp] = interp1(episode["timestamp"][frame_grasp - 10:frame_grasp])
        if frame_release < n_length:
            fr_end = min(n_length, frame_release + 10)
            interp = get_interp1d(episode["timestamp"][[frame_release, fr_end]], episode["gripper0_gripper_pose"][[frame_release, fr_end]])
            episode["gripper0_gripper_pose"][frame_release + 1:fr_end] = interp(episode["timestamp"][frame_release + 1:fr_end])
            if not single_arm:
                interp1 = get_interp1d(episode["timestamp"][[frame_release, fr_end]], episode["gripper1_gripper_pose"][[frame_release, fr_end]])
                episode["gripper1_gripper_pose"][frame_release + 1:fr_end] = interp1(episode["timestamp"][frame_release + 1:fr_end])

    episode["action"] = np.concatenate([
        episode["robot0_eef_pos"], episode["robot0_eef_rot_axis_angle"], episode["gripper0_gripper_pose"],
    ], axis=-1)
    if not single_arm:
        episode["action"] = np.concatenate([
            episode["action"], episode["robot1_eef_pos"], episode["robot1_eef_rot_axis_angle"], episode["gripper1_gripper_pose"],
        ], axis=-1)

    return episode


# ─────────────────────────────────────────────────────────────────────────────
# LeRobot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _import_lerobot(lerobot_src_path: str):
    if lerobot_src_path not in sys.path:
        sys.path.insert(0, lerobot_src_path)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import VideoEncodingManager
    return LeRobotDataset, VideoEncodingManager


def build_features(single_arm: bool, H: int, W: int, gripper_type: str = 'inspire_hand', vr_hand_dim: int = 0) -> dict:
    if gripper_type == 'gripper_f285':
        g0_names = ["gripper0_q"]
        g1_names = ["gripper1_q"]
        gripper_dim = 1
    else:
        g0_names = ["gripper0_p0", "gripper0_p1", "gripper0_p2", "gripper0_p3", "gripper0_p4", "gripper0_p5"]
        g1_names = ["gripper1_p0", "gripper1_p1", "gripper1_p2", "gripper1_p3", "gripper1_p4", "gripper1_p5"]
        gripper_dim = 6
    state_names = [
        "robot0_eef_pos_x", "robot0_eef_pos_y", "robot0_eef_pos_z",
        "robot0_eef_rot_x", "robot0_eef_rot_y", "robot0_eef_rot_z",
    ] + g0_names
    action_names = list(state_names)
    if not single_arm:
        extra = [
            "robot1_eef_pos_x", "robot1_eef_pos_y", "robot1_eef_pos_z",
            "robot1_eef_rot_x", "robot1_eef_rot_y", "robot1_eef_rot_z",
        ] + g1_names
        state_names = state_names + extra
        action_names = action_names + extra
    dim = (6 + gripper_dim) if single_arm else 2 * (6 + gripper_dim)
    features = {
        # all observation fields prefixed with "observation." per MotionTransDataset schema
        "observation.images.camera0":              {"dtype": "video",    "shape": (H, W, 3),       "names": ["height", "width", "channels"]},
        "observation.robot0_eef_pos":              {"dtype": "float32",  "shape": (3,),             "names": ["x", "y", "z"]},
        "observation.robot0_eef_rot_axis_angle":   {"dtype": "float32",  "shape": (3,),             "names": ["rx", "ry", "rz"]},
        "observation.gripper0_gripper_pose":       {"dtype": "float32",  "shape": (gripper_dim,),   "names": g0_names},
        "observation.camera0_pose":                {"dtype": "float32",  "shape": (6,),             "names": ["x", "y", "z", "rx", "ry", "rz"]},
        "observation.is_human":                    {"dtype": "float32",  "shape": (1,),             "names": None},
        # action stays unprefixed (standard LeRobot convention)
        "action":                                  {"dtype": "float32",  "shape": (dim,),           "names": action_names},
    }
    if not single_arm:
        features["observation.robot1_eef_pos"]            = {"dtype": "float32", "shape": (3,),           "names": ["x", "y", "z"]}
        features["observation.robot1_eef_rot_axis_angle"] = {"dtype": "float32", "shape": (3,),           "names": ["rx", "ry", "rz"]}
        features["observation.gripper1_gripper_pose"]     = {"dtype": "float32", "shape": (gripper_dim,), "names": g1_names}
    if vr_hand_dim > 0:
        _vr_joint_names = [
            "palm", "wrist",
            "thumb1", "thumb2", "thumb3", "thumb_tip",
            "index0", "index1", "index2", "index3", "index_tip",
            "middle0", "middle1", "middle2", "middle3", "middle_tip",
            "ring0", "ring1", "ring2", "ring3", "ring_tip",
            "pinky0", "pinky1", "pinky2", "pinky3", "pinky_tip",
        ]
        _axes = ["x", "y", "z", "euler_x", "euler_y", "euler_z"]
        n_joints = vr_hand_dim // 6
        if n_joints <= len(_vr_joint_names):
            hand_names = [f"{_vr_joint_names[i]}_{ax}" for i in range(n_joints) for ax in _axes]
        else:
            hand_names = [f"joint{i}_{ax}" for i in range(n_joints) for ax in _axes]
        features["observation.raw_vr_left_hand"]  = {"dtype": "float32", "shape": (vr_hand_dim,), "names": hand_names}
        features["observation.raw_vr_right_hand"] = {"dtype": "float32", "shape": (vr_hand_dim,), "names": hand_names}
        features["observation.raw_vr_head_pose"]  = {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "euler_x", "euler_y", "euler_z"]}
    return features



def conversion_lerobot_trajectory(
    input_data_fp_list: list,
    calib_quest2camera: np.ndarray,
    speed_downsample_ratio,
    single_arm: bool,
    hand_shrink_coef,
    gripper_type: str,
    f285_close_ramp_k: int,
    mode: str,
    out_resolutions_resize,
    out_resolutions_crop,
    resolution_image_final,
    num_points_final: int,
    points_max_distance_final: float,
    network_delay_checking: float,
    dataset,
    task_instruction: str,
    fps: int,
    n_demos: int | None = None,
    save_raw_vr: bool = False,
):
    pbar = tqdm(input_data_fp_list, desc="Episodes", total=len(input_data_fp_list))
    for save_dir, source, source_idx in pbar:
        if n_demos is not None and dataset.num_episodes >= n_demos:
            pbar.write(f"Reached n_demos={n_demos}, stopping.")
            break
        episode = conversion_single_trajectory(
            mode=mode,
            save_dir=save_dir,
            calib_quest2camera=calib_quest2camera,
            speed_downsample_ratio=speed_downsample_ratio,
            single_arm=single_arm,
            hand_shrink_coef=hand_shrink_coef,
            gripper_type=gripper_type,
            f285_close_ramp_k=f285_close_ramp_k,
            save_raw_vr=save_raw_vr,
            out_resolutions_resize=out_resolutions_resize,
            out_resolutions_crop=out_resolutions_crop,
            out_resolutions_image_final=resolution_image_final,
            network_delay_checking=network_delay_checking,
            num_points_final=num_points_final,
            points_max_distance_final=points_max_distance_final,
        )

        if episode is None:
            pbar.write(f"Skipping {save_dir}: no data or delay exceeded.")
            continue

        T = len(episode["timestamp"])
        if T == 0 or "camera0_rgb" not in episode:
            pbar.write(f"Skipping {save_dir}: empty episode or missing RGB.")
            continue

        for t in range(T):
            frame = {
                "observation.images.camera0":            (episode["camera0_rgb"][t] / 255.0).astype(np.float32),
                "observation.robot0_eef_pos":            episode["robot0_eef_pos"][t].astype(np.float32),
                "observation.robot0_eef_rot_axis_angle": episode["robot0_eef_rot_axis_angle"][t].astype(np.float32),
                "observation.gripper0_gripper_pose":     episode["gripper0_gripper_pose"][t].astype(np.float32),
                "observation.camera0_pose":              episode["camera0_pose"][t].astype(np.float32),
                "observation.is_human":                  np.array([1.0], dtype=np.float32),
                "action":                                episode["action"][t].astype(np.float32),
            }
            if not single_arm:
                frame["observation.robot1_eef_pos"]            = episode["robot1_eef_pos"][t].astype(np.float32)
                frame["observation.robot1_eef_rot_axis_angle"] = episode["robot1_eef_rot_axis_angle"][t].astype(np.float32)
                frame["observation.gripper1_gripper_pose"]     = episode["gripper1_gripper_pose"][t].astype(np.float32)
            if save_raw_vr and "raw_vr_left_hand" in episode:
                frame["observation.raw_vr_left_hand"]  = episode["raw_vr_left_hand"][t]
                frame["observation.raw_vr_right_hand"] = episode["raw_vr_right_hand"][t]
                frame["observation.raw_vr_head_pose"]  = episode["raw_vr_head_pose"][t]
            dataset.add_frame(frame, task=task_instruction, timestamp=float(t) / fps)

        dataset.save_episode()
        pbar.write(f"Saved episode {dataset.num_episodes - 1} from {save_dir} (T={T})")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--input_dir", "-i", required=True)
@click.option("--output", "-o", required=True)
@click.option("--calib_quest2camera_file", "-cf", required=True)
@click.option("--adapt_config_file", "-acf", default=None, type=str)
@click.option("--single_arm", is_flag=True, default=False)
@click.option("--default_speed_downsample_ratio", "-dsdr", default=1.0, type=float)
@click.option("--default_hand_shrink_coef", "-dhsc", default=1.0, type=float)
@click.option("--mode", "-m", required=True,
              type=click.Choice(["o", "p", "s", "a"], case_sensitive=False), default="o",
              help="o: only image (recommended for LeRobot)")
@click.option("--resolution_resize", "-ror", default="640x480")
@click.option("--resolution_crop", "-or", default="640x480")
@click.option("--resolution_image_final", "-for", default="224x224")
@click.option("--num_use_source", "-nus", default=None, type=int)
@click.option("--n_demos", "-nd", default=None, type=int,
              help="Max number of demos to convert per task. Default: convert all.")
@click.option("--num_points_final", "-npf", type=int, default=2048)
@click.option("--points_max_distance_final", "-pmdf", type=float, default=1.0)
@click.option("--n_encoding_threads", "-ne", default=-1, type=int,
              help="Ignored (LeRobot write is single-threaded). Kept for CLI compatibility.")
@click.option("--network_delay_checking", "-dl", default=0.5, type=float)
@click.option("--gripper_type", "-gt", default="inspire_hand",
              type=click.Choice(["inspire_hand", "gripper_f285"], case_sensitive=False),
              help="inspire_hand: 6-DOF Inspire Hand; gripper_f285: 1-DOF Robotiq 2F-85 [0, 0.8].")
@click.option("--f285_close_ramp_k", "-rk", default=5, type=int,
              help="F285 only: ramp k frames before each hand-close event from 0 → 0.8. 0 = disabled.")
@click.option("--save_raw_vr", "-srv", is_flag=True, default=False,
              help="Save raw VR hand joint poses and head pose (euler, VR world frame) in the dataset.")
@click.option("--repo_id", default="human_demo/task", type=str,
              help="LeRobot repo_id prefix; per-task instruction suffix appended automatically.")
@click.option("--fps", default=30, type=int,
              help="Dataset fps. Should match Quest rate / speed_downsample_ratio.")
@click.option("--lerobot_src_path", default="/Users/lochathien/Documents/Code/vr_lfd/src",
              type=str, help="Path to lerobot src directory for import.")
def main(
    input_dir, output, calib_quest2camera_file,
    adapt_config_file, single_arm,
    default_speed_downsample_ratio, default_hand_shrink_coef,
    mode,
    resolution_resize, resolution_crop, resolution_image_final,
    num_use_source, n_demos, num_points_final, points_max_distance_final,
    n_encoding_threads, network_delay_checking,
    gripper_type, f285_close_ramp_k, save_raw_vr,
    repo_id, fps, lerobot_src_path,
):
    out_resolution_resize = tuple(int(x) for x in resolution_resize.split("x"))
    out_resolution_crop = tuple(int(x) for x in resolution_crop.split("x"))
    resolution_image_final_tup = tuple(int(x) for x in resolution_image_final.split("x"))
    W, H = resolution_image_final_tup

    if n_encoding_threads > 1:
        print(f"[Warning] --n_encoding_threads={n_encoding_threads} ignored; LeRobot write is single-threaded.")

    if mode != "o":
        print(f"[Info] mode='{mode}': only camera0_rgb is written to LeRobot; depth/stereo/pointcloud are ignored.")

    LeRobotDataset, VideoEncodingManager = _import_lerobot(lerobot_src_path)

    adapt_config = {}
    if adapt_config_file is not None:
        with open(adapt_config_file) as f:
            adapt_config = json.load(f)

    calib_quest2camera = np.load(calib_quest2camera_file)

    input_dir_list = sorted(
        os.path.join(input_dir, x) for x in os.listdir(input_dir)
        if not x.startswith(".") and os.path.isdir(os.path.join(input_dir, x))
    )

    for task_dir in input_dir_list:
        input_folder = os.path.basename(task_dir)
        parts = input_folder.split("_")
        embodiment = parts[0]
        assert embodiment == "human", f"Expected 'human' folder prefix, got '{embodiment}'"
        environment_setting = parts[1]
        instruction = "_".join(parts[2:])

        task_repo_id = f"{repo_id}/{instruction}"
        task_output_dir = Path(os.path.expanduser(output)) / task_repo_id

        task_speed_ratio = default_speed_downsample_ratio
        task_shrink_coef = default_hand_shrink_coef
        if input_folder in adapt_config:
            p = adapt_config[input_folder]
            task_speed_ratio = p.get("speed_downsample_ratio", default_speed_downsample_ratio)
            task_shrink_coef = p.get("hand_shrink_coef", default_hand_shrink_coef)

        print(f"\nTask: {input_folder}")
        print(f"  Instruction:            {instruction}")
        print(f"  Repo ID:                {task_repo_id}")
        print(f"  Output:                 {task_output_dir}")
        print(f"  speed_downsample_ratio: {task_speed_ratio}")
        print(f"  hand_shrink_coef:       {task_shrink_coef}")

        def _list_dirs(path):
            return sorted(
                x for x in os.listdir(path)
                if not x.startswith(".") and os.path.isdir(os.path.join(path, x))
            )

        if environment_setting == "me":
            input_data_fp_list = []
            source_list = _list_dirs(task_dir)
            for sidx, source in enumerate(source_list):
                if num_use_source is not None and sidx >= num_use_source:
                    break
                src_path = os.path.join(task_dir, source)
                input_data_fp_list.extend(
                    (os.path.join(src_path, fp), source, sidx)
                    for fp in _list_dirs(src_path)
                )
        else:
            input_data_fp_list = [
                (os.path.join(task_dir, fp), "default", 0)
                for fp in _list_dirs(task_dir)
            ]

        if task_output_dir.exists():
            shutil.rmtree(task_output_dir)

        vr_hand_dim = _peek_vr_hand_dim(input_data_fp_list) if save_raw_vr else 0
        features = build_features(single_arm, H, W, gripper_type, vr_hand_dim)
        dataset = LeRobotDataset.create(
            repo_id=task_repo_id,
            fps=fps,
            features=features,
            root=Path(os.path.expanduser(output)),
            robot_type="human_vr",
            use_videos=True,
            tolerance_s=1.0 / fps,
            image_writer_processes=4,
            image_writer_threads=8,
        )

        with VideoEncodingManager(dataset):
            conversion_lerobot_trajectory(
                input_data_fp_list=input_data_fp_list,
                calib_quest2camera=calib_quest2camera,
                speed_downsample_ratio=task_speed_ratio,
                single_arm=single_arm,
                hand_shrink_coef=task_shrink_coef,
                gripper_type=gripper_type,
                f285_close_ramp_k=f285_close_ramp_k,
                mode=mode,
                out_resolutions_resize=out_resolution_resize,
                out_resolutions_crop=out_resolution_crop,
                resolution_image_final=resolution_image_final_tup,
                num_points_final=num_points_final,
                points_max_distance_final=points_max_distance_final,
                network_delay_checking=network_delay_checking,
                dataset=dataset,
                task_instruction=instruction,
                fps=fps,
                n_demos=n_demos,
                save_raw_vr=save_raw_vr,
            )

        print(f"Done: {dataset.num_episodes} episodes saved to {task_output_dir}")


if __name__ == "__main__":
    main()
