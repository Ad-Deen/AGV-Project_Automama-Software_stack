import vpi
import cv2
import cupy as cp
import numpy as np

class JetsonCameraHandler:
    def __init__(self, sensor_id, warp_map, width=1280, height=720, framerate=30, flip_method=2):
        self.sensor_id = sensor_id
        self.warp_map = warp_map
        self.stream = vpi.Stream()
        self.cap = None

        # Build GStreamer pipeline string for CSI camera
        self.pipeline = (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.sensor_id} with GStreamer pipeline.")

    def capture_and_process(self):
        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            return None

        try:
            frame_np = np.asarray(frame_bgr)
        
            with vpi.Backend.CUDA:
                with self.stream:
                    vpi_img = vpi.asimage(frame_np)
                    rectified = vpi_img.remap(self.warp_map).convert(vpi.Format.RGB8)
                    # self.stream.sync()
                    rectified_cpu = rectified.cpu()

                    return rectified_cpu

        except Exception as e:
            print(f"Error during capture_and_process: {e}")
            return None

    def close(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None  # Prevent future access

    def __del__(self):
        self.close()
