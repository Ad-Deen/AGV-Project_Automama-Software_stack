import cv2
import numpy as np
from ultralytics import YOLO

# Load the TensorRT-optimized segmentation model
model = YOLO("yolo11s-seg.engine", task="segment")

# Predict using streaming inference
results = model.predict(source="killo_road.mp4", stream=True)

# Generate distinct colors for up to 256 classes
def get_class_color_map(num_classes):
    np.random.seed(42)
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]

class_colors = get_class_color_map(len(model.names))

for r in results:
    frame = r.orig_img.copy()

    if r.masks is not None and r.masks.data is not None and r.boxes.cls is not None:
        masks = r.masks.data.detach().cpu().numpy()  # shape: [N, H, W]
        classes = r.boxes.cls.detach().cpu().numpy().astype(int)

        # Blank canvas for all instance masks (no overlay)
        combined_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        for i, cls_idx in enumerate(classes):
            color = class_colors[cls_idx]
            binary_mask = (masks[i] > 0.5).astype(np.uint8)

            # Resize mask to match frame dimensions if necessary
            if binary_mask.shape != frame.shape[:2]:
                binary_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Apply class-specific color
            for c in range(3):
                combined_mask[:, :, c] = np.where(binary_mask == 1, color[c], combined_mask[:, :, c])

        # Resize combined mask vertically
        combined_mask_resized = cv2.resize(combined_mask, (frame.shape[1], frame.shape[0] * 2), interpolation=cv2.INTER_NEAREST)

        # Display original frame and class-colored mask
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Segmentation Mask (Height x2)", combined_mask_resized)

    else:
        blank_mask = np.zeros((frame.shape[0]*2, frame.shape[1], 3), dtype=np.uint8)
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Segmentation Mask (Height x2)", blank_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
