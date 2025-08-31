import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
HFOV = 62.2
VFOV = 48.8


def compute_ground_depth_from_mask(mask, camera_height=0.9, vfov_deg=VFOV, hfov_deg=HFOV, pitch_deg=10.0):
    H, W = mask.shape
    
    # --- Vertical Angles (as before) ---
    degrees_per_pixel_v = vfov_deg / H
    center_row = H // 2
    pixel_rows = np.arange(H)
    angle_offsets_v = (center_row - pixel_rows) * degrees_per_pixel_v
    total_angles_v_deg = pitch_deg + angle_offsets_v
    total_angles_v_rad = np.deg2rad(np.clip(total_angles_v_deg, 1, 89))
    
    # Compute vertical ground distances
    ground_y = camera_height * np.tan(total_angles_v_rad)  # Shape: (H,)
    
    # --- Horizontal Angles ---
    degrees_per_pixel_h = hfov_deg / W
    center_col = W // 2
    pixel_cols = np.arange(W)
    angle_offsets_h_deg = (pixel_cols - center_col) * degrees_per_pixel_h
    angle_offsets_h_rad = np.deg2rad(angle_offsets_h_deg)  # Shape: (W,)
    
    # Compute cos(theta) for horizontal angles
    cos_theta = np.cos(angle_offsets_h_rad)  # Shape: (W,)
    
    # Expand vertical distances to full image
    y_map = np.tile(ground_y[:, np.newaxis], (1, W))  # Shape: (H, W)
    cos_map = np.tile(cos_theta[np.newaxis, :], (H, 1))  # Shape: (H, W)
    
    # Calculate true depth along x-axis (optical axis)
    true_depth_map = y_map / cos_map
    true_depth_map *= mask  # Keep only masked values
    
    return true_depth_map

def plot_occupancy_grid_from_depth(depth_map, mask, hfov_deg=HFOV, grid_width=100, grid_height=100, resolution=0.5, title="Occupancy Grid Map"):
    H, W = depth_map.shape

    # Horizontal FOV per pixel
    degrees_per_pixel_h = hfov_deg / W
    center_col = W // 2
    pixel_cols = np.arange(W)
    angle_offsets_deg = (pixel_cols - center_col) * degrees_per_pixel_h
    angle_offsets_rad = np.deg2rad(angle_offsets_deg)

    # Get depth values for valid mask points
    rows, cols = np.where((mask > 0) & (depth_map > 0))
    depths = depth_map[rows, cols]
    angles = angle_offsets_rad[cols]

    # Compute real-world coordinates
    X = depths * np.sin(angles)   # Left/right
    Y = depths * np.cos(angles)   # Forward

    # Initialize occupancy grid (0 = unknown/occupied, 255 = free)
    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

    # Convert X, Y to grid coordinates
    grid_x = (X / resolution + grid_width // 2).astype(int)
    grid_y = (Y / resolution).astype(int)

    # Filter points inside grid boundaries
    valid = (grid_x >= 0) & (grid_x < grid_width) & (grid_y >= 0) & (grid_y < grid_height)
    grid_x = grid_x[valid]
    grid_y = grid_y[valid]

    # Set free (road) areas to white
    grid[grid_height - 1 - grid_y, grid_x] = 255  # Invert Y-axis for display

    ## Plot the occupancy grid
    # plt.figure(figsize=(8, 10))
    # plt.imshow(grid, cmap='gray', origin='upper')
    # plt.title(title)
    # plt.xlabel("X (meters) — Horizontal")
    # plt.ylabel("Y (meters) — Forward")
    # plt.grid(True)
    # plt.show()

    return grid


class RoadMaskListener(Node):
    def __init__(self):
        super().__init__('road_mask_listener')

        self.bridge = CvBridge()
        self.mask = None
        # self.HFOV = 54.5/2
        # self.VFOV = 41.5
        self.depth_map = None
        self.grid = None
        self.grid_scale = 5

        # Subscribe to the /road_mask topic
        self.subscription = self.create_subscription(
            Image,
            '/road_mask',
            self.road_mask_callback,
            10)

        self.get_logger().info("Road mask listener node started.")

    def road_mask_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV grayscale image
            self.mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            if self.mask is None or len(self.mask.shape) != 2:
                self.get_logger().warn("Invalid or empty mask received.")
                return
            
            self.depth_map = compute_ground_depth_from_mask(self.mask)
            
            if self.depth_map is None or np.max(self.depth_map) == 0:
                self.get_logger().warn("Depth map is empty or zero.")
                return
            self.grid = plot_occupancy_grid_from_depth(self.depth_map,self.mask)
            # Normalize for display
            # depth_display = np.clip(self.depth_map / np.max(self.depth_map) * 255, 0, 255).astype(np.uint8)

            resized_grid = cv2.resize(self.grid, (self.grid.shape[1] * self.grid_scale, self.grid.shape[0] * self.grid_scale), interpolation=cv2.INTER_NEAREST)
            # cv2.imshow("Depth map", depth_display)
            cv2.imshow("Occupancy grid", resized_grid)
            cv2.waitKey(1)


        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RoadMaskListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
