import cupy as cp
import cv2
import numpy as np

class ColorMappedDepthMapCreator:
    """
    A GPU-accelerated class to create a color-mapped depth map from a road mask.

    This class pre-computes the necessary trigonometric maps once during
    initialization, allowing for fast, repeated computation of the depth map
    from new road masks without recalculating the static camera geometry.
    """
    def __init__(self, camera_height_cm, vfov_deg, hfov_deg, pitch_deg, image_width, image_height):
        """
        Initializes the creator with camera parameters and pre-computes the
        trigonometric maps on the GPU.

        Args:
            camera_height_cm (float): Camera height from the ground in centimeters.
            vfov_deg (float): Vertical field of view in degrees.
            hfov_deg (float): Horizontal field of view in degrees.
            pitch_deg (float): Camera's pitch angle in degrees.
            image_width (int): Width of the camera image.
            image_height (int): Height of the camera image.
        """
        self.H = image_height
        self.W = image_width
        self.camera_height_cm = cp.array(camera_height_cm, dtype=cp.float32)

        with cp.cuda.Device(0):
            # --- 1. Vertical Angle Pre-computation (Base Distance) ---
            # Calculate the vertical angle for each pixel row, relative to the horizon.
            degrees_per_pixel_v = vfov_deg / self.H
            center_row = self.H // 2
            pixel_rows = cp.arange(self.H, dtype=cp.float32)
            angle_offsets_v = -(center_row - pixel_rows) * degrees_per_pixel_v
            total_angles_v_deg = cp.array(pitch_deg) + angle_offsets_v
            total_angles_v_rad = cp.deg2rad(cp.clip(total_angles_v_deg, 1, 89))
            # print(total_angles_v_deg)
            # Compute the base ground distance for each pixel row assuming the central column.
            ground_y_cm = self.camera_height_cm/cp.tan(total_angles_v_rad)
            self.y_map_cm = cp.tile(ground_y_cm[:, cp.newaxis], (1, self.W))
            # print(self.y_map_cm)
            # --- 2. Horizontal Angle Pre-computation (Distance Adjustment) ---
            # Calculate the horizontal angle for each pixel column.
            degrees_per_pixel_h = hfov_deg / self.W
            center_col = self.W // 2
            pixel_cols = cp.arange(self.W, dtype=cp.float32)
            angle_offsets_h_rad = cp.deg2rad((pixel_cols - center_col) * degrees_per_pixel_h)
            
            # Pre-compute the cosine map used to adjust the base distance for horizontal offset.
            self.cos_map = cp.tile(cp.cos(angle_offsets_h_rad)[cp.newaxis, :], (self.H, 1))

    def create_color_mapped_depth_map(self, mask_cp):
        """
        Computes a color-mapped depth map from a binary road mask.

        This method uses the pre-computed maps to efficiently calculate the
        depth map and return a color-coded visualization.

        Args:
            mask_cp (cupy.ndarray): A 2D binary CuPy array (H, W) where non-zero
                                    values represent road pixels.

        Returns:
            numpy.ndarray: A 3-channel BGR NumPy array (H, W, 3) representing the
                           color-mapped ground depth.
        """
        with cp.cuda.Device(0):
            # --- 1. Compute the Final Depth Map on the GPU ---
            # Adjust the pre-computed vertical distance (`y_map_cm`) by the
            # horizontal angle to find the true depth for every pixel.
            true_depth_map_cm = (self.y_map_cm / 100) / self.cos_map
            # print(true_depth_map_cm[:,320])
            print(f"min:{cp.min(true_depth_map_cm)}m, max: {cp.max(true_depth_map_cm)}m")
            # Apply the original mask to the depth map.
            true_depth_map_cm = true_depth_map_cm * mask_cp
            # print(f"min:{cp.min(true_depth_map_cm)}m, max: {cp.max(true_depth_map_cm)}m")
            # --- 2. Normalize and Color Map the Depth Map ---
            # Get the numerical depth map from GPU to CPU
            depth_map_np = true_depth_map_cm.get()
            # print(depth_map_np[:,320])
            
            # Normalize depth values to a standard range (0-255) for color mapping
            max_depth = np.max(depth_map_np)
            # print(max_depth)
            if max_depth > 0:
                normalized_depth = (depth_map_np / max_depth * 255).astype(np.uint8)
                # print(normalized_depth)
            else:
                normalized_depth = np.zeros_like(depth_map_np, dtype=np.uint8)
            
            # Apply a color map
            color_mapped_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

            # Ensure non-road pixels remain black
            color_mapped_depth[depth_map_np == 0] = [0, 0, 0]
            
        return color_mapped_depth
