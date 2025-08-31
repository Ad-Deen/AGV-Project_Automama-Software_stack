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

            # --- 3. Initialize the CuPy kernel once
            self._initialize_kernel()

    def _initialize_kernel(self):
        """
        Initializes and compiles the CuPy RawKernel for costmap inflation.
        This is called once during class initialization.
        """
        # Define the kernel code as a string
        self.inflation_kernel = cp.RawKernel(r'''
            extern "C" __global__ void inflate_costmap(unsigned char* costmap, int rows, int cols, int h_step, int v_step) {
                // Get the current thread's global index
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                // Check if the thread is within the grid boundaries
                if (x < cols && y < rows) {
                    int index = y * cols + x;
                    
                    // Only process pixels that are initially free space (0)
                    if (costmap[index] == 0) {
                        bool found_free_space = false;
                        
                        // --- Vertical check (upward only)
                        for (int i = 1; i <= 7; ++i) {
                            if (y - i >= 0) {
                                // Check for a pixel that is not an obstacle (i.e., < 250)
                                if (costmap[index - (i+v_step) * cols] == 0) {
                                    found_free_space = true;
                                    break; // Stop searching once free space is found
                                }
                            } else {
                                break; // Out of bounds
                            }
                        }
                        
                        // Assign value based on the search result
                        if (!found_free_space) {
                            costmap[index] = 200; // Did not find free space upward, so this pixel is inflated
                        }
                    }
                }
            }
            ''', 'inflate_costmap')



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

    def create_fixed_birdseye_view_depth_map(self, depth_map_cp,
                                        grid_width_px=500,
                                        grid_height_px=400,
                                        v_pixels_per_meter=10,
                                        h_pixels_per_meter=40,
                                        ):

        with cp.cuda.Device(0):
            road_mask_cp = depth_map_cp > 0
            if cp.sum(road_mask_cp) == 0:
                empty_map = cp.zeros((grid_height_px, grid_width_px, 3), dtype=cp.uint8)
                # Return the empty visualization AND the empty raw grid
                return empty_map.get(), None

            road_y_m = self.y_map_m[road_mask_cp]  # forward distance (m)
            road_x_m = self.x_map_m[road_mask_cp]  # left/right (m)
            road_depth_m = depth_map_cp[road_mask_cp]
            #------- Check lengths with these parameters -----------------
            # print(f"max:{cp.max(road_depth_m)}m min: {cp.min(road_depth_m)}")
            # print(f"y max:{cp.max(road_y_m)}m y min: {cp.min(road_y_m)}")
            
            # This is how you print the full contents of the CuPy array
            # We convert it to a NumPy array on the CPU for a detailed printout
            # print("Full road_y_m array (on CPU):")
            # print(road_y_m.get())

            # Vehicle pixel location in BEV grid
            vehicle_x_px = grid_width_px // 2
            vehicle_y_px = grid_height_px - 1  # bottom row

            # Convert meters to pixel coordinates relative to vehicle
            grid_x_pixels = vehicle_x_px + (road_x_m * h_pixels_per_meter)
            grid_y_pixels = vehicle_y_px - (road_y_m * v_pixels_per_meter)

            # Round and convert to int indices
            grid_x_pixels = cp.rint(grid_x_pixels).astype(cp.int32)
            grid_y_pixels = cp.rint(grid_y_pixels).astype(cp.int32)

            # Filter points outside the grid bounds
            valid_mask = (grid_x_pixels >= 0) & (grid_x_pixels < grid_width_px) & \
                        (grid_y_pixels >= 0) & (grid_y_pixels < grid_height_px)

            grid_x_pixels = grid_x_pixels[valid_mask]
            grid_y_pixels = grid_y_pixels[valid_mask]
            road_depth_m = road_depth_m[valid_mask]

            # Initialize BEV grid
            bev_grid = cp.zeros((grid_height_px, grid_width_px), dtype=cp.float32)

            # Plot depths into BEV grid (if multiple points land on same pixel, later ones overwrite)
            bev_grid[grid_y_pixels, grid_x_pixels] = road_depth_m

            # Draw vehicle marker: small bright square at vehicle pixel pos
            marker_size = 10
            y_start = max(vehicle_y_px - marker_size, 0)
            y_end = min(vehicle_y_px + marker_size, grid_height_px - 1)
            x_start = max(vehicle_x_px - marker_size, 0)
            x_end = min(vehicle_x_px + marker_size, grid_width_px - 1)

            # Do not draw the marker on the original bev_grid, as it will affect costmap calculation
            # bev_grid[y_start:y_end, x_start:x_end] = 1.0  # bright marker
            # print(bev_grid)

        #---------------Only for visuals--------------------------------------
            # Normalize and color map
            max_depth = cp.max(bev_grid)
            if max_depth > 0:
                normalized_bev = (bev_grid / max_depth * 255).astype(cp.uint8)
            else:
                normalized_bev = cp.zeros_like(bev_grid, dtype=cp.uint8)

            normalized_bev_np = normalized_bev.get()
            color_mapped_bev = cv2.applyColorMap(normalized_bev_np, cv2.COLORMAP_JET)

            # Apply a specific color to the vehicle marker region AFTER color mapping
            color_mapped_bev[y_start:y_end, x_start:x_end] = [0, 255, 0] # Green marker

            # Black background for zero-depth pixels
            color_mapped_bev[normalized_bev_np == 0] = [0, 0, 0]

            return color_mapped_bev , bev_grid

    def create_dynamic_costmap(self, bev_grid,
                                h_step=20, v_step=10):
            """
            Generates a costmap with a binary obstacle layer and a hardline cost layer
            for vehicle clearance using a manual, GPU-based approach.
            
            Args:
                bev_grid (cupy.ndarray): A 2D CuPy array representing the occupancy grid
                                        with depth values in meters.
                h_step (int): The horizontal step size for checking neighbors.
                v_step (int): The vertical step size for checking neighbors.
        
            Returns:
                cupy.ndarray: A 2D CuPy array representing the costmap, with values
                        from 0 to 255.
            """
            with cp.cuda.Device(0):
                if bev_grid is None:
                    return None

                # 1. Create a binary costmap (0=free, 255=occupied)
                costmap = (bev_grid == 0).astype(cp.uint8) * 255
                
                # Get the dimensions of the costmap
                rows, cols = costmap.shape
                
                # Set up the grid and block dimensions for the kernel
                # This determines how many threads will be used
                block_size = (16, 16)
                grid_size = (
                    (cols + block_size[0] - 1) // block_size[0],
                    (rows + block_size[1] - 1) // block_size[1]
                )

                # Launch the pre-compiled kernel on the costmap array
                self.inflation_kernel(grid_size, block_size, (costmap, rows, cols, h_step, v_step))
                
                return costmap.astype(cp.uint8)
