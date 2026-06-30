import time
from argparse import ArgumentParser
import numpy as np
import os
import pickle
import cv2
from human_data.keyboard_quest_recorder import KeyboardQuestRecorder
try:
    from human_data.camera_zed_simple import CameraZedSimple
except Exception:
    CameraZedSimple = None
from human_data.camera_realsense_simple import CameraLiteRealSense
from common.precise_sleep import precise_wait
from common.timestamp_accumulator import ObsAccumulator


def get_zed_camera(args, verbose=False, add_record=False):
    resolution = (1280, 720)
    num_threads = 2
    print("VideoStereoRecorder Initialization completed")
    serial_number_list = CameraZedSimple.get_connected_devices_serial()
    print(serial_number_list)
    device_id = serial_number_list[0]
    camera = CameraZedSimple(
        device_id=device_id,
        camera_exposure=args.camera_exposure,
        resolution=resolution,
        capture_fps=args.frequency,
        num_threads=num_threads,
        recording=add_record,
        recording_crop_w=args.mp4_crop_w,
        recording_crop_h=args.mp4_crop_h,
        recording_downsample_ratio=args.mp4_downsample_ratio,
        verbose=verbose
    )
    return camera, device_id


def get_realsense_camera(args, verbose=False):
    resolution = (args.mp4_crop_w, args.mp4_crop_h) if (args.mp4_crop_w and args.mp4_crop_h) else (1280, 720)
    print("RealSense Initialization")
    serial_number_list = CameraLiteRealSense.get_connected_devices_serial()
    print(serial_number_list)
    device_id = serial_number_list[0]
    camera = CameraLiteRealSense(
        device_id=device_id,
        resolution=resolution,
        capture_fps=args.frequency,
    )
    return camera, device_id


def start_record(args, quest, camera, device_id):
    save_dir = quest.data_dir
    video_path = os.path.join(save_dir, "rgb.mp4")
    with open(os.path.join(save_dir, "device_id.txt"), "w") as f:
        f.write(device_id)
    if not args.no_camera:
        if hasattr(camera, 'get_intrinsic_left_cam'):
            intrinsic = camera.get_intrinsic_left_cam()
            np.save(os.path.join(save_dir, "camera_intrinsic.npy"), intrinsic)
        camera.start_recording(video_path=str(video_path))
    action_accumulator = ObsAccumulator()
    return action_accumulator, save_dir


def save_episode(save_dir, action_accumulator, left_hand):
    timestamps = np.array(action_accumulator.timestamps['actions'])
    if len(timestamps) > 5:
        dt_check = timestamps[5:] - timestamps[4:-1]
        dt_check_max = np.max(dt_check)
        if dt_check_max > 0.12:
            print(f"Warning: time interval too large ({dt_check_max:.3f}s > 0.12), check camera/network")
            with open(os.path.join(save_dir, "dt_check_unlimited.txt"), "w") as f:
                f.write(f"The time interval / network delay between two frames is too large ({dt_check_max} secs, > 0.12), please check the camera or your network connection")
        with open(os.path.join(save_dir, "dt_check_max.txt"), "w") as f:
            f.write(f"Max time interval / network delay: {dt_check_max} secs")
        print("Max dt:", dt_check_max)
    len_hand_arr = 6 * len(left_hand.hand_pose)
    episode = {
        'timestamp': timestamps,
        'left_hand_mat': np.array(action_accumulator.data['actions'])[..., :len_hand_arr],
        'right_hand_mat': np.array(action_accumulator.data['actions'])[..., len_hand_arr:2 * len_hand_arr],
        'head_pose_mat': np.array(action_accumulator.data['actions'])[..., 2 * len_hand_arr:],
    }
    with open(os.path.join(save_dir, 'episode.pkl'), 'wb') as f:
        pickle.dump(episode, f)
    print(f"[SAVED] Episode saved to {save_dir}")


def main(args):
    camera = None
    device_id = "no_camera"

    if not args.no_camera:
        if args.camera == 'realsense':
            camera, device_id = get_realsense_camera(args, verbose=(not args.no_verbose))
        else:
            camera, device_id = get_zed_camera(args, verbose=(not args.no_verbose), add_record=(not args.no_mp4_record))
        camera.recieve()

    os.makedirs(args.output_dir, exist_ok=True)
    quest = KeyboardQuestRecorder(args.output_dir)
    print("Recorder Initialization completed")
    print("Step 1: Do ONE left-middle-pinch on Quest to start hand data streaming.")
    print("Step 2: Press [SPACE] in the camera window to start/stop each episode. [Q] to quit.")

    # blank window for keyboard capture when no camera
    if args.no_camera:
        blank = np.zeros((100, 400, 3), dtype=np.uint8)
        cv2.putText(blank, "SPACE: start/stop  Q: quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('control', blank)

    action_accumulator = None
    save_dir = None
    recording = False
    last_hand = None
    dt = 1 / args.frequency

    while True:
        try:
            t_fps_start = time.time()
            t_cycle_end = t_fps_start + dt

            status, xrhand, head_pose, timestamp = quest.receive(verbose=False)

            # keyboard detection via opencv
            key = cv2.waitKey(1) & 0xFF
            space_pressed = (key == ord(' '))
            quit_pressed = (key == ord('q') or key == 27)

            if quit_pressed:
                print("Quit requested.")
                break

            if space_pressed:
                if not recording:
                    quest.start_recording_manual()
                    action_accumulator, save_dir = start_record(args, quest, camera, device_id)
                    recording = True
                    print("[REC] Recording started... Press [SPACE] to stop and save.")
                else:
                    quest.stop_recording_manual()
                    if not args.no_camera:
                        camera.stop_recording()
                    if save_dir is not None and action_accumulator is not None and last_hand is not None:
                        save_episode(save_dir, action_accumulator, last_hand)
                    else:
                        print("[WARN] No data accumulated, episode not saved.")
                    recording = False
                    action_accumulator = None
                    save_dir = None
                    last_hand = None
                    quest.data_dir = None

            if status == "Data" and head_pose is not None and recording:
                if not args.no_camera:
                    img = camera.recieve()
                    if img is not None:
                        cv2.imshow('camera', img[..., ::-1])
                left_hand, right_hand = xrhand
                last_hand = left_hand
                left_hand_arr = left_hand.get_hand_6d_pose_array()
                right_hand_arr = right_hand.get_hand_6d_pose_array()
                head_pose_arr = head_pose.return_6d_pose()
                actions = np.concatenate([left_hand_arr, right_hand_arr, head_pose_arr])
                action_accumulator.put(
                    data={"actions": actions[None]},
                    timestamps=np.array([timestamp])
                )
            elif not args.no_camera and not recording:
                img = camera.recieve()
                if img is not None:
                    cv2.imshow('camera', img[..., ::-1])

            precise_wait(t_cycle_end, time_func=time.time)
            if not args.no_verbose:
                t_now = time.time()
                rec_label = "REC" if recording else status
                print(f"Human Data Collection Keyboard FPS: {1 / (t_now - t_fps_start):.1f}  [{rec_label}]")

        except KeyboardInterrupt:
            if not args.no_camera:
                camera.stop_recording()
                camera.close()
            quest.close()
            break
        except Exception as e:
            if not args.no_camera:
                camera.stop_recording()
            quest.close()
            raise ValueError(e)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("--camera", type=str, default='zed', choices=['zed', 'realsense'])
    parser.add_argument("-z", "--camera_exposure", type=int, default=None)
    parser.add_argument("--no_mp4_record", action="store_true", default=False)
    parser.add_argument("--no_verbose", action="store_true", default=False)
    parser.add_argument("--no_camera", action="store_true", default=False)
    parser.add_argument("--mp4_crop_w", type=int, default=None)
    parser.add_argument("--mp4_crop_h", type=int, default=None)
    parser.add_argument("--mp4_downsample_ratio", type=int, default=2)
    args = parser.parse_args()
    if not os.path.isdir("data"):
        os.mkdir("data")
    main(args)

# python human_data_collection_keyboard.py -o data/data_human_raw --mp4_crop_w 640 --mp4_crop_h 480 --mp4_downsample_ratio 2
