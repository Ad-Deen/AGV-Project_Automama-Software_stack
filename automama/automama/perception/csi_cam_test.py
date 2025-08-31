import cv2

def gstreamer_pipeline(
    sensor_id=1,
    capture_width=640,
    capture_height=480,
    framerate=30,
    flip_method=2,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        # f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
    )


def main():
    print("Using GStreamer pipeline:")
    pipeline = gstreamer_pipeline()
    print(cv2.getBuildInformation())
    print(pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ Failed to open CSI camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame capture failed, stopping...")
            break

        cv2.imshow("CSI Camera Stream", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
