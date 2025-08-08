import cv2
import time
from trt_infer_utils import TensorRTInference

def main():
    engine_path = "/home/deen/ros2_ws/src/automama/automama/perception/PID net models/engine/pidnet_L_480x640_32f.engine"
    video_path = "/home/deen/ros2_ws/src/automama/automama/perception/killo_road.mp4"

    # with TensorRTInference(engine_path) as trt_model:
    inf = TensorRTInference(engine_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return

    frame_idx = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        mask = inf.infer(frame)
        dt = time.time() - start_time
        total_time += dt

        blended = cv2.addWeighted(frame, 0.2, mask, 0.8, 0)
        fps_info = f"FPS: {1.0 / dt:.1f} | Frame: {frame_idx}"
        cv2.putText(blended, fps_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Segmented", blended)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nAvg FPS: {frame_idx / total_time:.2f}")

if __name__ == "__main__":
    main()
