import pyrealsense2 as rs
import numpy as np
import imageio
import time
import os


class CameraLiteRealSense:

    def __init__(self, device_id, resolution=(1280, 720), capture_fps=30):
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.device_id = device_id
        self.video_writer = None
        self._timestamps = []
        self._video_path = None
        w, h = self.resolution
        fps = self.capture_fps
        self.align = rs.align(rs.stream.color)
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

        try:
            rs_config.enable_device(self.device_id)

            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)

            d = pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, 1)

            color_stream = pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ['fx', 'fy', 'ppx', 'ppy', 'height', 'width']
            self.intrinsics_array = np.zeros(len(order))
            for i, name in enumerate(order):
                self.intrinsics_array[i] = getattr(intr, name)

            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            self.pipeline = pipeline
        except Exception as e:
            print(f"[LiteRealsense {self.device_id}] Error: {e}")
            rs_config.disable_all_streams()

    @staticmethod
    def get_connected_devices_serial():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == 'D400':
                    serials.append(serial)
        serials = sorted(serials)
        return serials

    def get_video_stream(self):
        try:
            frameset = self.pipeline.wait_for_frames()
            frameset = self.align.process(frameset)
            color_frame = np.asarray(frameset.get_color_frame().get_data())[:, :, ::-1]  # bgr to rgb
            depth_frame = np.asarray(frameset.get_depth_frame().get_data())
            return color_frame, depth_frame
        except Exception as e:
            print(f"[LiteRealsense {self.device_id}] Fail to get Frame. Error: {e}")
            return None, None

    def recieve(self):
        color_frame, _ = self.get_video_stream()
        if color_frame is not None and self.video_writer is not None:
            self._timestamps.append(time.time())
            self.video_writer.append_data(color_frame)
        return color_frame

    def start_recording(self, video_path: str):
        self._timestamps = []
        self._video_path = video_path
        self.video_writer = imageio.get_writer(video_path, format='ffmpeg', fps=self.capture_fps)
        print(f"[RealSense {self.device_id}] Recording started.")
        return True

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
            if self._video_path is not None:
                ts_path = os.path.join(os.path.dirname(self._video_path), "camera_timestamps.npy")
                np.save(ts_path, np.array(self._timestamps))
        print(f"[RealSense {self.device_id}] Recording stopped.")

    def close(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()

    def get_intrinsic(self):
        fx, fy, ppx, ppy = self.intrinsics_array[:4]
        mat = np.eye(3)
        mat[0, 0] = fx
        mat[1, 1] = fy
        mat[0, 2] = ppx
        mat[1, 2] = ppy
        return mat

    def get_intrinsic_left_cam(self):
        return self.get_intrinsic()

    def get_intrinsic_left_dist(self):
        return np.zeros(5)
