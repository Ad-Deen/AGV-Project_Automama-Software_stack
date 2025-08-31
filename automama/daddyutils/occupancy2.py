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
            degrees_per_pixel_v = vfov_deg / self.H
            center_row = self.H // 2
            pixel_rows = cp.arange(self.H, dtype=cp.float32)
            angle_offsets_v = -(center_row - pixel_rows) * degrees_per_pixel_v
            total_angles_v_deg = cp.array(pitch_deg) + angle_offsets_v
            total_angles_v_rad = cp.deg2rad(cp.clip(total_angles_v_deg, 1, 89))
            ground_y_cm = self.camera_height_cm/cp.tan(total_angles_v_rad)
            self.y_map_m = cp.tile(ground_y_cm[:, cp.newaxis], (1, self.W))/100

            # --- 2. Horizontal Angle Pre-computation (Distance Adjustment) ---
            degrees_per_pixel_h = hfov_deg / self.W
            center_col = self.W // 2
            pixel_cols = cp.arange(self.W, dtype=cp.float32)
            angle_offsets_h_rad = cp.deg2rad((pixel_cols - center_col) * degrees_per_pixel_h)
            
            self.cos_map = cp.tile(cp.cos(angle_offsets_h_rad)[cp.newaxis, :], (self.H, 1))
            self.tan_map = cp.tile(cp.tan(angle_offsets_h_rad)[cp.newaxis, :], (self.H, 1))
            # print(self.tan_map.shape)
            self.x_map_m = (self.y_map_m * self.tan_map)
            # print(f"max:{cp.max(self.x_map_m)}m min: {cp.min(self.x_map_m)}")
            # print(self.x_map_m.shape)

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
            # Adjust the pre-computed vertical distance (`y_map_m`) by the
            # horizontal angle to find the true depth for every pixel.
            true_depth_map_m = (self.y_map_m) / self.cos_map 
            true_depth_map_m = true_depth_map_m * mask_cp
            depth_map_cp = cp.where(true_depth_map_m > 50.0, 0.0, true_depth_map_m)

            # print(f"max:{cp.max(depth_map_cp)}m min: {cp.min(depth_map_cp)}")
            # print(true_depth_map_m.shape)
            # --- 2. Normalize and Color Map the Depth Map ---
            # Get the numerical depth map from GPU to CPU
            depth_map_np = true_depth_map_m.get()
            
        return depth_map_np,  depth_map_cp

    def create_birdseye_view_depth_map(self, depth_map_cp, grid_padding_pixels=10, v_pixels_per_meter=10, h_pixels_per_meter=40):
        """
        Generates a Bird's-Eye View (BEV) map of the road depth from an existing
        depth map.

        Args:
            depth_map_cp (cupy.ndarray): A 2D CuPy array (H, W) of depth values in meters
                                        (where 0 indicates a non-road pixel).
            grid_padding_pixels (int): Padding to add to the BEV grid's edges.
            pixels_per_meter (int): Scaling factor for meters to pixels on the BEV grid.

        Returns:
            numpy.ndarray: A 3-channel BGR NumPy array (H, W, 3) of the BEV map.
        """
        with cp.cuda.Device(0):
            # 1. Get coordinates for road pixels
            road_mask_cp = depth_map_cp > 0
            if cp.sum(road_mask_cp) == 0:
                # Return an empty map if no road pixels are found.
                empty_map = cp.zeros((400, 400, 3), dtype=cp.uint8)
                return empty_map.get()

            # Get the y and x world coordinates for the road pixels.
            # The original maps are in m, so we convert to meters here.
            road_y_m = (self.y_map_m)[road_mask_cp]
            road_x_m = (self.x_map_m)[road_mask_cp]
            road_depth_m = depth_map_cp[road_mask_cp]

            #------- Check lengths with these parameters -----------------
            # print(f"max:{cp.max(road_depth_m)}m min: {cp.min(road_depth_m)}")
            print(f"y max:{cp.max(road_y_m)}m y min: {cp.min(road_y_m)}")
            print(f"x max:{cp.max(road_x_m)}m x min: {cp.min(road_x_m)}")

            # 2. Determine the dynamic grid size in pixels, centered on the vehicle.
            # The grid width will be from the furthest left to furthest right road points.
            min_road_x_m, max_road_x_m = cp.min(road_x_m), cp.max(road_x_m)
            grid_width_m = max_road_x_m - min_road_x_m
            grid_width_pixels = int(cp.ceil(grid_width_m * h_pixels_per_meter)) + 2 * grid_padding_pixels

            # The grid height will be from the vehicle's position (y=0) to the furthest road point.
            max_road_y_m = cp.max(road_y_m)
            grid_height_m = max_road_y_m
            grid_height_pixels = int(cp.ceil(grid_height_m * v_pixels_per_meter)) + 2 * grid_padding_pixels

            # 3. Create the BEV grid and populate it
            bev_grid = cp.zeros((grid_height_pixels, 500), dtype=cp.float32)

            # Convert world coordinates (meters) to grid coordinates (pixels).
            # The grid's x-origin (0) will correspond to min_road_x_m.
            # The grid's y-origin (0) will correspond to max_road_y_m (the furthest point).
            grid_y_pixels = (max_road_y_m - road_y_m) * v_pixels_per_meter + grid_padding_pixels
            grid_x_pixels = (road_x_m - min_road_x_m) * h_pixels_per_meter + grid_padding_pixels

            # Plot the depth values onto the BEV grid using direct indexing.
            bev_grid[grid_y_pixels.astype(cp.int32), grid_x_pixels.astype(cp.int32)] = road_depth_m

            # 4. Plot the vehicle marker at the bottom-center of the grid.
            # We define a small square for the marker.
            marker_size = 5
            marker_y = grid_height_pixels - grid_padding_pixels
            marker_x = grid_width_pixels // 2
            # marker_x = 500 // 2
            # The value 1.0 will make the marker appear bright in the color-mapped image.
            bev_grid[marker_y - marker_size:marker_y + marker_size,
                    marker_x - marker_size:marker_x + marker_size] = 1.0

            # 5. Normalize and color map the BEV grid
            max_depth = cp.max(bev_grid)
            if max_depth > 0:
                normalized_bev = (bev_grid / max_depth * 255).astype(cp.uint8)
            else:
                normalized_bev = cp.zeros_like(bev_grid, dtype=cp.uint8)

            # Convert to NumPy for OpenCV color mapping
            normalized_bev_np = normalized_bev.get()
            color_mapped_bev = cv2.applyColorMap(normalized_bev_np, cv2.COLORMAP_JET)

            # Ensure the color-mapped BEV still has a black background for non-road areas
            color_mapped_bev[normalized_bev_np == 0] = [0, 0, 0]

            return color_mapped_bev

