"""
Human data recorder using MCAP format.
Replaces data_teleop_server.py + quest_recorder.py combo.

Records Quest hand tracking and camera frames into a single .mcap file per episode.
Calibration (quest2camera) is embedded directly in the MCAP — no external .npy needed.

Usage:
    python -m human_data.mcap_data_recorder \
        --calib_path camera_params/quest_realsense/calib_result_quest2camera.npy \
        --output_dir data/mcap_sessions \
        --frequency 20 \
        --camera realsense

Controls (via Quest app):
    Start  → begin recording
    Save   → save episode and stop
    Cancel → discard episode
"""

import os
import time
import datetime
import argparse
import numpy as np

from human_data.quest_recorder import QuestRecorder
from human_data.mcap_recorder import MCAPRecorder
from human_data.camera_realsense_simple import CameraLiteRealSense

try:
    from human_data.camera_zed_simple import CameraZedSimple
except ImportError:
    CameraZedSimple = None


def make_output_path(output_dir: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{ts}.mcap")


def main(args):
    # --- load calibration ---
    quest2camera = np.load(args.calib_path)
    assert quest2camera.shape == (4, 4), f"Expected (4,4), got {quest2camera.shape}"
    print(f"Loaded quest2camera from {args.calib_path}")

    # --- init camera ---
    camera = None
    if not args.no_camera:
        if args.camera == "realsense":
            serials = CameraLiteRealSense.get_connected_devices_serial()
            assert serials, "No RealSense device found."
            camera = CameraLiteRealSense(serials[0], resolution=(1280, 720), capture_fps=args.frequency)
        elif args.camera == "zed":
            assert CameraZedSimple is not None, "ZED SDK not installed."
            serials = CameraZedSimple.get_connected_devices_serial()
            assert serials, "No ZED device found."
            camera = CameraZedSimple(serials[0], resolution=(1280, 720))
        else:
            raise ValueError(f"Unknown camera: {args.camera}")
        intrinsics = camera.get_intrinsic()
        print("Camera initialized.")
    else:
        intrinsics = np.eye(3)  # placeholder when no camera

    # --- init Quest receiver ---
    quest = QuestRecorder(output_dir="/tmp/quest_tmp")
    print("Quest receiver initialized. Waiting for Quest app...")

    recorder: MCAPRecorder = None
    current_ts = time.time()
    dt = 1.0 / args.frequency

    while True:
        now = time.time()
        if now - current_ts < dt:
            continue
        current_ts = now

        # --- get camera frame ---
        rgb_frame = None
        if camera is not None:
            rgb_frame, _ = camera.get_video_stream()

        # --- receive Quest data ---
        status, xrhand, head_pose, timestamp = quest.receive(verbose=args.verbose)

        if status == "Wait" or status == "Wait-Ensure":
            continue

        if status == "Start":
            output_path = make_output_path(args.output_dir)
            recorder = MCAPRecorder(
                output_path=output_path,
                quest2camera=quest2camera,
                intrinsics=intrinsics,
                jpeg_quality=args.jpeg_quality,
            )
            recorder.start()
            print(f"Recording started → {output_path}")
            continue

        if status in ("Save", "Cancel", "Ensure"):
            if recorder is not None:
                recorder.close()
                if status == "Cancel":
                    os.remove(recorder.output_path)
                    print(f"Episode cancelled, file removed.")
                else:
                    print(f"Episode saved: {recorder.output_path}")
                recorder = None
            continue

        if status == "Data" and recorder is not None:
            left_hand, right_hand = xrhand
            recorder.record_quest(left_hand, right_hand, head_pose, timestamp=timestamp)
            if rgb_frame is not None:
                recorder.record_camera(rgb_frame, timestamp=time.time())

    if camera is not None:
        camera.close()
    quest.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_path", type=str, required=True,
                        help="Path to calib_result_quest2camera.npy")
    parser.add_argument("--output_dir", type=str, default="data/mcap_sessions")
    parser.add_argument("--camera", type=str, default="realsense", choices=["realsense", "zed", "none"])
    parser.add_argument("--no_camera", action="store_true")
    parser.add_argument("--frequency", type=int, default=20)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.camera == "none":
        args.no_camera = True
    main(args)
