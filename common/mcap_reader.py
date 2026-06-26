"""
MCAP reader for human VR data — drop-in counterpart to SVOReader + episode.pkl.

Returns data in the same format that zarr_human_data_conversion.py expects:
  episode_org = {
      "left_hand_mat":  np.ndarray (T, 26*6),   6D euler poses per joint
      "right_hand_mat": np.ndarray (T, 26*6),
      "head_pose_mat":  np.ndarray (T, 6),
      "timestamp":      np.ndarray (T,),          Unix seconds
  }
Plus camera frame access matching SVOReader.read_camera() interface.
"""

import json
import numpy as np
import cv2
from typing import Optional, Dict, Tuple
from mcap.reader import make_reader

from human_data.mcap_recorder import (
    TOPIC_LEFT_HAND, TOPIC_RIGHT_HAND, TOPIC_HEAD, TOPIC_CAMERA_RGB,
    ATTACH_CALIB, ATTACH_INTRINSICS,
)


def _ns_to_sec(t_ns: int) -> float:
    return t_ns / 1e9


class MCAPReader:
    """
    Reads a .mcap file written by MCAPRecorder.

    Attributes:
        quest2camera:  (4, 4) float64 — embedded calibration matrix
        intrinsics:    (3, 3) float64 — camera intrinsic matrix
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.quest2camera: Optional[np.ndarray] = None
        self.intrinsics: Optional[np.ndarray] = None

        # raw timeline lists, filled by _load()
        self._left_hand_poses = []    # list of (t_sec, arr(156,))
        self._right_hand_poses = []
        self._head_poses = []         # list of (t_sec, arr(6,))
        self._camera_frames = []      # list of (t_sec, np.ndarray HxWx3)

        self._load()
        self._camera_idx = 0          # pointer for read_camera()

    def _load(self):
        with open(self.filepath, "rb") as f:
            reader = make_reader(f)

            # --- calibration attachments ---
            summary = reader.get_summary()
            if summary is not None:
                for att in (summary.attachment_indexes or []):
                    pass  # indexes only, need to seek; handled below

            # seek attachments via iter_attachments if available, else fallback
            f.seek(0)
            reader2 = make_reader(f)
            try:
                for attachment in reader2.iter_attachments():
                    if attachment.name == ATTACH_CALIB:
                        self.quest2camera = np.frombuffer(
                            attachment.data, dtype=np.float64
                        ).reshape(4, 4)
                    elif attachment.name == ATTACH_INTRINSICS:
                        meta = json.loads(attachment.data.decode())
                        K = np.eye(3, dtype=np.float64)
                        K[0, 0] = meta["fx"]
                        K[1, 1] = meta["fy"]
                        K[0, 2] = meta["cx"]
                        K[1, 2] = meta["cy"]
                        self.intrinsics = K
            except AttributeError:
                pass  # older mcap versions may not have iter_attachments

            # --- messages ---
            f.seek(0)
            reader3 = make_reader(f)
            for schema, channel, message in reader3.iter_messages():
                topic = channel.topic
                t = _ns_to_sec(message.log_time)
                if topic == TOPIC_LEFT_HAND:
                    arr = np.array(json.loads(message.data.decode()), dtype=np.float64)
                    self._left_hand_poses.append((t, arr))
                elif topic == TOPIC_RIGHT_HAND:
                    arr = np.array(json.loads(message.data.decode()), dtype=np.float64)
                    self._right_hand_poses.append((t, arr))
                elif topic == TOPIC_HEAD:
                    arr = np.array(json.loads(message.data.decode()), dtype=np.float64)
                    self._head_poses.append((t, arr))
                elif topic == TOPIC_CAMERA_RGB:
                    buf = np.frombuffer(message.data, dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    self._camera_frames.append((t, rgb))

    # ------------------------------------------------------------------
    # Episode dict interface (matches episode.pkl used by conversion script)
    # ------------------------------------------------------------------

    def load_episode(self) -> Dict[str, np.ndarray]:
        """Return dict matching episode.pkl format, keyed by Quest timestamps."""
        if not self._left_hand_poses:
            raise RuntimeError("No Quest data found in MCAP file.")

        timestamps = np.array([t for t, _ in self._left_hand_poses])
        left_mat = np.stack([arr for _, arr in self._left_hand_poses])
        right_mat = np.stack([arr for _, arr in self._right_hand_poses])
        head_mat = np.stack([arr for _, arr in self._head_poses])

        return {
            "timestamp": timestamps,
            "left_hand_mat": left_mat,
            "right_hand_mat": right_mat,
            "head_pose_mat": head_mat,
        }

    # ------------------------------------------------------------------
    # Camera frame interface (matches SVOReader used by conversion script)
    # ------------------------------------------------------------------

    def get_frame_count(self) -> int:
        return len(self._camera_frames)

    def get_frame_resolution(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        if not self._camera_frames:
            return (0, 0)
        h, w = self._camera_frames[0][1].shape[:2]
        return (w, h)

    def get_camera_timestamps(self) -> np.ndarray:
        """Unix seconds for each camera frame."""
        return np.array([t for t, _ in self._camera_frames])

    def get_camera_information(self) -> dict:
        """Matches SVOReader.get_camera_information() return format."""
        return {
            "left_intrinsic": self.intrinsics,
            "right_intrinsic": self.intrinsics,   # monocular: same both sides
            "stereo_transform": np.eye(4),
        }

    def read_camera(self, return_timestamp: bool = False):
        """
        Read next camera frame. Matches SVOReader.read_camera() interface.
        Returns None at end of file.
        """
        if self._camera_idx >= len(self._camera_frames):
            return None
        t, rgb = self._camera_frames[self._camera_idx]
        self._camera_idx += 1

        serial = "mcap"
        data_dict = {
            "image": {
                serial + "_left": rgb,
                serial + "_right": rgb,
            }
        }
        if return_timestamp:
            return data_dict, int(t * 1e3)   # milliseconds, matches SVO convention
        return data_dict

    def set_frame_index(self, index: int):
        self._camera_idx = index

    def disable_camera(self):
        pass  # nothing to release for MCAP
