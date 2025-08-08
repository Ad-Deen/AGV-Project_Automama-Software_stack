import cv2
import os

# ==== CONFIG ====
SAVE_DIR = "automama/perception/stereo_vision_test/single_cam_KD_callib/captured_images"  # Change as needed
FLIP_METHOD = 2           # 0 = none, 2 = 180°, etc.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
CHECKERBOARD = (8, 6)     # (Columns-1, Rows-1) inner corners of (8x6) checkerboard

def gstreamer_pipeline(sensor_id=0, width=960, height=576, fps=30, flip_method=2):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(
        gstreamer_pipeline(width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=FPS, flip_method=FLIP_METHOD),
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():
        print("❌ Failed to open CSI camera.")
        return

    print("📷 Press 's' to check and save if checkerboard is found. Press 'q' to quit.")

    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame capture failed.")
            break
        frame_disp = cv2.resize(frame,(900,600))
        # Display the live frame
        cv2.imshow("Camera Stream", frame_disp)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)

            if found:
                saved_frame_count += 1
                filename = os.path.join(SAVE_DIR, f"frame_{saved_frame_count:04d}.png")
                cv2.imwrite(filename, frame)
                print(f"✅ Checkerboard found! Saved frame #{saved_frame_count} to {filename}")
                
                # Optional: visualize corners
                frame_vis = frame.copy()
                cv2.drawChessboardCorners(frame_vis, CHECKERBOARD, corners, found)
                cv2.imshow("Detected Checkerboard", frame_vis)
                cv2.waitKey(500)  # Pause to show result
                cv2.destroyWindow("Detected Checkerboard")
            else:
                print("⚠️ Checkerboard NOT found. Frame skipped.")

        elif key == ord('q'):
            print("👋 Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
