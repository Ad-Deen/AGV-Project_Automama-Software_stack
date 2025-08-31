import cv2
import os
import time

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    framerate=30,
    flip_method=2,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
    )

def get_unique_filenames(base_dir, base_name="Vehicle_mounted", ext=".mp4"):
    """Return unique filenames for left and right: Vehicle_mounted_left_1.mp4, Vehicle_mounted_right_1.mp4, ..."""
    i = 1
    while True:
        left_file = f"{base_name}_left_{i}{ext}"
        right_file = f"{base_name}_right_{i}{ext}"
        left_path = os.path.join(base_dir, left_file)
        right_path = os.path.join(base_dir, right_file)
        if not (os.path.exists(left_path) or os.path.exists(right_path)):
            return left_path, right_path
        i += 1

def main():
    capture_width = 640
    capture_height = 480
    framerate = 30
    clip_duration = 60  # seconds per clip

    # Open left (sensor-id=0) and right (sensor-id=1) cameras
    pipeline_left = gstreamer_pipeline(sensor_id=0, capture_width=capture_width, capture_height=capture_height, framerate=framerate)
    pipeline_right = gstreamer_pipeline(sensor_id=1, capture_width=capture_width, capture_height=capture_height, framerate=framerate)

    cap_left = cv2.VideoCapture(pipeline_left, cv2.CAP_GSTREAMER)
    cap_right = cv2.VideoCapture(pipeline_right, cv2.CAP_GSTREAMER)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Failed to open one or both CSI cameras.")
        return

    base_dir = "/home/deen/ros2_ws/src/automama/automama/perception/saved_vids"
    os.makedirs(base_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        while True:
            # Create new files for stereo pair
            left_path, right_path = get_unique_filenames(base_dir)
            out_left = cv2.VideoWriter(left_path, fourcc, framerate, (capture_width, capture_height))
            out_right = cv2.VideoWriter(right_path, fourcc, framerate, (capture_width, capture_height))
            print(f"Recording new stereo clips:\n  Left : {left_path}\n  Right: {right_path}")

            start_time = time.time()

            while True:
                ret_left, frame_left = cap_left.read()
                ret_right, frame_right = cap_right.read()

                if not ret_left or not ret_right:
                    print("Error: Frame capture failed from one or both cameras.")
                    break

                out_left.write(frame_left)
                out_right.write(frame_right)

                # Combine preview (side-by-side)
                preview = cv2.hconcat([frame_left, frame_right])
                cv2.imshow("Stereo Preview (Left | Right)", preview)

                # Stop recording if 'q' pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap_left.release()
                    cap_right.release()
                    out_left.release()
                    out_right.release()
                    cv2.destroyAllWindows()
                    print("Recording stopped and resources released.")
                    return

                # Stop this clip after clip_duration and start new one
                if time.time() - start_time >= clip_duration:
                    break

            out_left.release()
            out_right.release()
            print("Finished saving stereo clip.")

    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
