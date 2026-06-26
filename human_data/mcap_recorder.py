"""
MCAP-based recorder for human VR data collection.
Replaces the combination of episode.pkl + recording.svo2 with a single .mcap file.

Topics:
  /quest/left_hand    - JSON, 26 joints × 6 floats (euler pose), Quest frequency
  /quest/right_hand   - same
  /quest/head         - JSON, 6 floats (euler pose)
  /camera/rgb         - raw bytes (JPEG), camera frequency

Attachments (static, written once):
  calib/quest2camera  - 4×4 float64 matrix as raw bytes
  camera/intrinsics   - JSON {"fx","fy","cx","cy"}
"""

import json
import time
import struct
import numpy as np
import cv2
from mcap.writer import Writer

from human_data.models import XRHand, Transform


TOPIC_LEFT_HAND = "/quest/left_hand"
TOPIC_RIGHT_HAND = "/quest/right_hand"
TOPIC_HEAD = "/quest/head"
TOPIC_CAMERA_RGB = "/camera/rgb"
ATTACH_CALIB = "calib/quest2camera"
ATTACH_INTRINSICS = "camera/intrinsics"


def _sec_to_ns(t: float) -> int:
    return int(t * 1e9)


def _encode_pose_array(arr: np.ndarray) -> bytes:
    """Encode flat float64 array as JSON bytes."""
    return json.dumps(arr.tolist()).encode()


def _encode_image(rgb: np.ndarray, quality: int = 90) -> bytes:
    """Encode RGB frame as JPEG bytes."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode camera frame")
    return buf.tobytes()


class MCAPRecorder:
    """
    Records Quest hand tracking + camera frames into a single .mcap file.

    Usage:
        recorder = MCAPRecorder(
            output_path="session/recording.mcap",
            quest2camera=np.load("camera_params/quest_realsense/calib_result_quest2camera.npy"),
            intrinsics=camera.get_intrinsic(),
        )
        recorder.start()
        while recording:
            # call at Quest frequency (e.g. 50 Hz)
            recorder.record_quest(left_hand, right_hand, head_pose)
            # call at camera frequency (e.g. 20-30 Hz)
            recorder.record_camera(rgb_frame)
        recorder.close()
    """

    def __init__(
        self,
        output_path: str,
        quest2camera: np.ndarray,
        intrinsics: np.ndarray,
        jpeg_quality: int = 90,
    ):
        assert quest2camera.shape == (4, 4), "quest2camera must be 4×4"
        assert intrinsics.shape == (3, 3), "intrinsics must be 3×3"

        self.output_path = output_path
        self.quest2camera = quest2camera
        self.intrinsics = intrinsics
        self.jpeg_quality = jpeg_quality

        self._file = None
        self._writer = None
        self._ch = {}   # topic -> channel_id
        self._schema_id = None
        self._seq = 0

    def start(self):
        self._file = open(self.output_path, "wb")
        self._writer = Writer(self._file)
        self._writer.start()

        # one shared schema: raw JSON
        self._schema_id = self._writer.register_schema(
            name="json", encoding="jsonschema", data=b"{}"
        )
        self._image_schema_id = self._writer.register_schema(
            name="jpeg", encoding="", data=b""
        )

        for topic in [TOPIC_LEFT_HAND, TOPIC_RIGHT_HAND, TOPIC_HEAD]:
            self._ch[topic] = self._writer.register_channel(
                topic=topic,
                message_encoding="json",
                schema_id=self._schema_id,
            )
        self._ch[TOPIC_CAMERA_RGB] = self._writer.register_channel(
            topic=TOPIC_CAMERA_RGB,
            message_encoding="",
            schema_id=self._image_schema_id,
        )

        # embed calibration and intrinsics as attachments
        self._writer.add_attachment(
            name=ATTACH_CALIB,
            media_type="application/octet-stream",
            data=self.quest2camera.astype(np.float64).tobytes(),
            create_time=0,
            log_time=0,
        )
        intr_meta = {
            "fx": float(self.intrinsics[0, 0]),
            "fy": float(self.intrinsics[1, 1]),
            "cx": float(self.intrinsics[0, 2]),
            "cy": float(self.intrinsics[1, 2]),
        }
        self._writer.add_attachment(
            name=ATTACH_INTRINSICS,
            media_type="application/json",
            data=json.dumps(intr_meta).encode(),
            create_time=0,
            log_time=0,
        )

    def record_quest(
        self,
        left_hand: XRHand,
        right_hand: XRHand,
        head_pose: Transform,
        timestamp: float = None,
    ):
        """Call at Quest polling frequency. timestamp is Unix time in seconds."""
        if timestamp is None:
            timestamp = time.time()
        t_ns = _sec_to_ns(timestamp)

        self._writer.add_message(
            channel_id=self._ch[TOPIC_LEFT_HAND],
            log_time=t_ns,
            publish_time=t_ns,
            sequence=self._seq,
            data=_encode_pose_array(left_hand.get_hand_6d_pose_array()),
        )
        self._writer.add_message(
            channel_id=self._ch[TOPIC_RIGHT_HAND],
            log_time=t_ns,
            publish_time=t_ns,
            sequence=self._seq,
            data=_encode_pose_array(right_hand.get_hand_6d_pose_array()),
        )
        self._writer.add_message(
            channel_id=self._ch[TOPIC_HEAD],
            log_time=t_ns,
            publish_time=t_ns,
            sequence=self._seq,
            data=_encode_pose_array(head_pose.return_6d_pose()),
        )
        self._seq += 1

    def record_camera(self, rgb: np.ndarray, timestamp: float = None):
        """Call at camera capture frequency. rgb is (H, W, 3) uint8 RGB."""
        if timestamp is None:
            timestamp = time.time()
        t_ns = _sec_to_ns(timestamp)
        self._writer.add_message(
            channel_id=self._ch[TOPIC_CAMERA_RGB],
            log_time=t_ns,
            publish_time=t_ns,
            sequence=self._seq,
            data=_encode_image(rgb, self.jpeg_quality),
        )

    def close(self):
        if self._writer is not None:
            self._writer.finish()
            self._writer = None
        if self._file is not None:
            self._file.close()
            self._file = None
