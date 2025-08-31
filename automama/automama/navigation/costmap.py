import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
HFOV = 54.5/2
VFOV = 41.5

def plot_occupancy_grid_from_depth(depth_map, mask, hfov_deg=HFOV, grid_width=150, grid_height=250, resolution=0.5, title="Occupancy Grid Map"):
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

    # Plot the occupancy grid
    plt.figure(figsize=(8, 10))
    plt.imshow(grid, cmap='gray', origin='upper')
    plt.title(title)
    plt.xlabel("X (meters) — Horizontal")
    plt.ylabel("Y (meters) — Forward")
    plt.grid(True)
    plt.show()

    return grid

def plot_depth_map_points(depth_map, mask, hfov_deg=HFOV, title="Depth Map Points"):
    H, W = depth_map.shape
    
    # Horizontal angles for each pixel col (degrees and radians)
    degrees_per_pixel_h = hfov_deg / W
    center_col = W // 2
    pixel_cols = np.arange(W)
    angle_offsets_h_deg = (pixel_cols - center_col) * degrees_per_pixel_h
    angle_offsets_h_rad = np.deg2rad(angle_offsets_h_deg)
    
    # Get indices of masked pixels with valid depth
    rows, cols = np.where((mask > 0) & (depth_map > 0))
    
    depths = depth_map[rows, cols]
    thetas = angle_offsets_h_rad[cols]
    
    # Compute X, Y coordinates in meters (camera coordinate system)
    X = depths * np.sin(thetas)
    Y = depths * np.cos(thetas)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, s=5, c='blue', alpha=0.7)
    plt.xlabel("X (meters) — Horizontal")
    plt.ylabel("Y (meters) — Forward")
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def compute_ground_depth_from_mask(mask, camera_height=1.0, vfov_deg=VFOV, hfov_deg=HFOV, pitch_deg=10.0):
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

# Load the .npy array
mask = np.load("/home/deen/ros2_ws/src/automama/automama/navigation/image_gray.npy")
depth_map = compute_ground_depth_from_mask(mask)
# occup = create_occupancy_grid_from_mask_and_depth(mask,depth_map)
plt.imshow(depth_map, cmap='plasma')
plt.title("Corrected Depth Map from Road Mask")
plt.colorbar(label="Depth (m)")
plt.axis('off')
plt.show()
plot_depth_map_points(depth_map,mask)
grid = plot_occupancy_grid_from_depth(depth_map,mask)
# np.set_printoptions(threshold=np.inf)  # Don't truncate large arrays
# print(depth_map[196, :])  # 
# cv2.imshow(depth_map)
