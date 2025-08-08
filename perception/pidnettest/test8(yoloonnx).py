import cv2
import numpy as np
import onnxruntime as ort

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (640, 640))
    img = frame_resized.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img

def run_segmentation(model_path, video_path):
    # Initialize ONNX Runtime session
    sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    print(f"Model outputs: {output_names}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess_frame(frame)
        outputs = sess.run(output_names, {input_name: inp})

        # Get segmentation map: output1 -> [1, 32, 160, 160]
        seg = outputs[1][0]  # shape: [32, 160, 160]
        seg_mask = np.argmax(seg, axis=0).astype(np.uint8)  # shape: [160, 160]

        # Resize to original size for display
        seg_mask_resized = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        seg_mask_colored = cv2.applyColorMap(seg_mask_resized * 8, cv2.COLORMAP_JET)

        blended = cv2.addWeighted(frame, 0.6, seg_mask_colored, 0.4, 0)

        cv2.imshow("Segmentation", blended)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
run_segmentation("/home/deen/ros2_ws/src/automama/automama/perception/best.onnx", "/home/deen/ros2_ws/src/automama/automama/perception/killo_road.mp4")
