import cupy as cp
import numpy as np
import cv2

# =================================================================================================
# ================================Euclidean gradieny function and kernel =============================
euclidean_gradient = """
extern "C" __global__
void euclidean_gradient_kernel(unsigned char* costmap,
                         int rows,
                         int cols,
                         int start_x,
                         int start_y
                         ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int index = y * cols + x;

        // Apply cost calculation only to pixels with a value less than 10
        if (costmap[index] < 250) {
            float dx = (float)x - start_x;
            float dy = (float)y - start_y;
            float dist_sq = dx*dx + dy*dy;
            float distance = sqrtf(dist_sq) ; // CORRECTED: This is the Euclidean distance

            float euclidean_cost = distance ;
            int new_cost = costmap[index] + (int)euclidean_cost * 0.1; //scaling factor = 1.0
            
            // Ensure the new cost does not exceed the maximum value of 245
            costmap[index] = (unsigned char)((new_cost < 245) ? new_cost : 245);
        }
    }
}
"""
def euclidean_costmap(costmap_cp):
    """
    Applies a custom CUDA kernel to add Euclidean-based cost to the costmap.

    The cost is calculated based on the distance from a static start point.

    Args:
        costmap_cp (cupy.ndarray): The input costmap as a CuPy array.

    Returns:
        cupy.ndarray: The modified costmap.
    """
    rows, cols = costmap_cp.shape

    # Check for empty costmap
    if rows == 0:
        return costmap_cp

    # Calculate the dynamic start position as the mid-center bottom pixel
    start_x = cols // 2
    start_y = rows - 1
    # euclidean_factor = 1.0
    # start_x, start_y = start_pos

    # Define the kernel launch configuration.
    threads_per_block = (16, 16)
    blocks_per_grid_x = (cols + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (rows + threads_per_block[1] - 1) // threads_per_block[1]

    # Load the new kernel
    cost_increase_kernel = cp.RawKernel(euclidean_gradient, 'euclidean_gradient_kernel')

    # Call the kernel with a 2D grid configuration and the new parameters.
    cost_increase_kernel(
        (blocks_per_grid_x, blocks_per_grid_y),
        threads_per_block,
        (costmap_cp, rows, cols, start_x, start_y)
    )

    return costmap_cp

# ==================================================================================
# ================================Euclidean gradient from TARGET function and kernel ==================
euclidean_gradient_target = """
extern "C" __global__
void euclidean_gradient_target_kernel(unsigned char* costmap,
                                     int rows,
                                     int cols,
                                     int target_x,
                                     int target_y
                                     ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int index = y * cols + x;
        // Apply cost calculation only to pixels with a value less than 10
        if (costmap[index] < 250) {
            float dx = (float)x - target_x;
            float dy = (float)y - target_y;
            float dist_sq = dx*dx + dy*dy;
            float distance = sqrtf(dist_sq); // This is the Euclidean distance

            // Note: We are using subtraction here to create a cost that decreases with distance to target
            // and an scaling factor of 0.5 to make it less prominent
            int new_cost = costmap[index] + (int)(distance) *0.1; 
            
            // Ensure the new cost does not exceed the maximum value of 245
            costmap[index] = (unsigned char)((new_cost < 245) ? new_cost : 245);
        }
    }
}
"""
def euclidean_costmap_from_target(costmap_cp, target_cp):
    """
    Applies a custom CUDA kernel to add a potential field from the target point.

    Args:
        costmap_cp (cupy.ndarray): The input costmap as a CuPy array.
        target_cp (cupy.ndarray): The target coordinates [y, x] as a CuPy array.

    Returns:
        cupy.ndarray: The modified costmap.
    """
    rows, cols = costmap_cp.shape

    # Check for empty costmap
    if rows == 0:
        return costmap_cp

    target_y, target_x = target_cp

    threads_per_block = (16, 16)
    blocks_per_grid_x = (cols + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (rows + threads_per_block[1] - 1) // threads_per_block[1]

    target_potential_kernel = cp.RawKernel(euclidean_gradient_target, 'euclidean_gradient_target_kernel')

    target_potential_kernel(
        (blocks_per_grid_x, blocks_per_grid_y),
        threads_per_block,
        (costmap_cp, rows, cols, int(target_x), int(target_y))
    )

    return costmap_cp
# ============================================================
#========================== Gap Follow =====================
# ------------------ CUDA Kernel (single-pass Bresenham per ray) ------------------
# CUDA kernel
_RAY_SCAN_KERNEL = r"""
extern "C" __global__
void angular_ray_scan(const unsigned char* __restrict__ dt,
                      const int H, const int W,
                      const int x0, const int y0,
                      const float max_angle_deg,
                      const int num_rays,
                      const int max_range,
                      int* __restrict__ out_dist)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rays) return;

    // Compute angle for this ray [-max_angle_deg, +max_angle_deg]
    float a_deg = (num_rays <= 1) ? 0.0f : -max_angle_deg + 2.0f * max_angle_deg * tid / (float)(num_rays - 1);
    float a_rad = a_deg * 3.14159265358979323846f / 180.0f;

    float sin_a = sinf(a_rad);
    float cos_a = cosf(a_rad);

    int steps = 0;

    for (int s = 1; s <= max_range; ++s) {
        int dx = (int)(s * sin_a);
        int dy = (int)roundf(s * cos_a);

        int x = x0 + dx;
        int y = y0 - dy; // upward in image coordinates

        // Bounds check
        if (x < 0 || x >= W || y < 0 || y >= H) break;

        unsigned char v = dt[y * W + x];

        if (v >= 130) break; // blocked
        steps += 1;         // free
    }

    out_dist[tid] = steps;
}
"""

# Compile kernel
_angular_ray_kernel = cp.RawKernel(_RAY_SCAN_KERNEL, "angular_ray_scan")


def scan_longest_gap(costmap: cp.ndarray,
                     start_row: int,
                     start_col: int = 250,
                     max_angle_deg: float = 89.0,
                     num_rays: int = 90,
                     max_range: int = 300,
                     prev_angle_deg = 45,
                     step_threshold = 3,
                     ):
    """
    Scan rays upward from (start_row, start_col) and return the longest free gap.
    
    Returns:
        best_len_px: int, length of longest free gap in pixels
        best_angle_deg: float, angle corresponding to that gap
    """
    if costmap.dtype != cp.uint8:
        costmap = costmap.astype(cp.uint8)
    H, W = costmap.shape

    # Allocate GPU array to store distances
    out_dist = cp.zeros((num_rays,), dtype=cp.int32)

    # Launch kernel
    tpb = 128
    blocks = (num_rays + tpb - 1) // tpb

    _angular_ray_kernel((blocks,), (tpb,), (
        costmap,
        np.int32(H), np.int32(W),
        np.int32(start_col), np.int32(start_row),
        np.float32(max_angle_deg),
        np.int32(num_rays),
        np.int32(max_range),
        out_dist
    ))

    # Copy distances to host
    v_dist = out_dist.get()
    # print(v_dist)
    # Compute angles for each ray
    angles_deg = np.linspace(-max_angle_deg, max_angle_deg, num_rays)

    # Find the longest free gap
    best_idx = int(np.argmax(v_dist))
    best_len_px = int(v_dist[best_idx])
    target_angle_deg = float(angles_deg[best_idx])

    # Smooth steering toward target
    delta = target_angle_deg - prev_angle_deg
    if abs(delta) > step_threshold:
        delta = np.sign(delta) * step_threshold

    smooth_angle_deg = prev_angle_deg + delta

    # Ensure within allowed range
    smooth_angle_deg = float(np.clip(smooth_angle_deg, -max_angle_deg, max_angle_deg))
    # prev_angle_deg = smooth_angle_deg
    # print(len(v_dist))
    # --- Step 4: Map smoothed angle to exact index ---
    # Convert angle to index (linear mapping)
    angle_fraction = (smooth_angle_deg + max_angle_deg) / (2 * max_angle_deg)
    exact_idx = int(round(angle_fraction * (num_rays - 1)))

    # --- Step 5: Gap length along smoothed angle ---
    smooth_len_px = int(v_dist[exact_idx])
    return smooth_len_px, smooth_angle_deg

def draw_direction_arrow(costmap: np.ndarray,
                         start_row,
                         start_col,
                         angle_deg: float,
                         gap_len: int,
                         color=(0, 255, 255),
                         thickness=2) -> np.ndarray:
    """
    Draw an arrow on a costmap showing the selected direction.

    Args:
        costmap    : np.ndarray (H,W,3) or (H,W) grayscale (will be converted to BGR)
        start_row  : starting pixel row (y) (int or scalar)
        start_col  : starting pixel col (x) (int or scalar)
        angle_deg  : direction angle in degrees, 0 = straight up (-y), +right = +x
        gap_len    : scalar length in pixels
        color      : BGR tuple for arrow color
        thickness  : arrow thickness

    Returns:
        img_out : copy of costmap with arrow drawn
    """
    # Ensure integer Python values
    start_row = int(cp.asnumpy(start_row)) if isinstance(start_row, cp.ndarray) else int(start_row)
    start_col = int(cp.asnumpy(start_col)) if isinstance(start_col, cp.ndarray) else int(start_col)

    # Make a color copy
    if len(costmap.shape) == 2:
        img_out = cv2.cvtColor(costmap, cv2.COLOR_GRAY2BGR)
    else:
        img_out = costmap.copy()

    # Compute end point
    angle_rad = np.deg2rad(angle_deg)
    dx = int(round(np.sin(angle_rad) * gap_len))
    dy = int(round(-np.cos(angle_rad) * gap_len))  # up is negative y in image

    end_col = start_col + dx
    end_row = start_row + dy

    # Clip to image bounds
    H, W = img_out.shape[:2]
    end_col = np.clip(end_col, 0, W - 1)
    end_row = np.clip(end_row, 0, H - 1)

    # Draw arrow
    cv2.arrowedLine(img_out,
                    (start_col, start_row),
                    (end_col, end_row),
                    color=color,
                    thickness=thickness,
                    tipLength=0.05)
    return img_out

# ================================Interpolation functions ==================
# Logic:
# 1. Each thread is assigned to a unique pixel (row, col).
# 2. The thread checks if its pixel is a non-zero value.
# 3. If the pixel is non-zero, it scans to its left (within the margin) for a zero.
# 4. It then scans to its right (within the margin) for a zero.
# 5. If it finds a zero on both sides, it means this pixel is "sandwiched"
#    between two zeros within the margin. The thread then sets its pixel to zero.
# 6. If the pixel is already zero, the thread does nothing.
add_zeros_right_kernel_code = """
extern "C" __global__ void add_zeros_right_kernel(
    const unsigned char* in_data,  // read-only
    unsigned char* out_data,       // write output
    int height,
    int width,
    int margin) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int idx = row * width + col;

        // Default: copy original value
        out_data[idx] = in_data[idx];

        if (in_data[idx] == 0) {
            int zero_end_col = -1;
            int max_scan = min(margin, width - col - 1);

            // Scan to the right for another zero
            for (int i = 1; i <= max_scan; ++i) {
                if (in_data[row * width + (col + i)] == 0) {
                    zero_end_col = col + i;
                    break;
                }
            }
            
            // If another zero was found, fill the gap with zeros
            if (zero_end_col != -1) {
                // The thread fills all pixels BETWEEN the two zeros.
                for (int i = col + 1; i < zero_end_col; ++i) {
                    out_data[row * width + i] = 0;
                }
            }
        }
    }
}
"""

def add_zeros_right_kernel_wrapper(image_cp, margin=2):
    """
    Launch the horizontal zero-fill kernel that fills between two zeros to the right.

    Args:
        image_cp (cupy.ndarray): The input 2D CuPy array of type uint8.
        margin (int): The maximum distance to scan right for a zero pixel.

    Returns:
        cupy.ndarray: A new CuPy array with the gaps filled.
    """
    if image_cp.dtype != cp.uint8:
        raise ValueError("Input array must be of type cupy.uint8.")
    if image_cp.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    
    height, width = image_cp.shape
    block_size = (16, 16)
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])

    kernel = cp.RawKernel(add_zeros_right_kernel_code, 'add_zeros_right_kernel')

    output_cp = image_cp.copy()  # safe output buffer
    kernel(
        grid_size, block_size,
        (image_cp, output_cp,
         np.int32(height), np.int32(width), np.int32(margin))
    )
    return output_cp



fill_zeros_upward_kernel_code = """
extern "C" __global__ void fill_zeros_upward_kernel(
    const unsigned char* in_data,
    unsigned char* out_data,
    int height,
    int width,
    int margin) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        unsigned char val = in_data[row * width + col];
        out_data[row * width + col] = val; // start with input value

        if (val == 0) {
            int upper_zero_row = -1;
            int max_scan = min(margin, row);
            for (int i = 1; i <= max_scan; ++i) {
                if (in_data[(row - i) * width + col] == 0) {
                    upper_zero_row = row - i;
                    break;
                }
            }
            if (upper_zero_row != -1) {
                for (int i = upper_zero_row + 1; i < row; ++i) {
                    out_data[i * width + col] = 0;
                }
            }
        }
    }
}

"""


def add_zeros_upward_kernel_wrapper(image_cp, margin=3):
    if image_cp.dtype != cp.uint8:
        raise ValueError("Input array must be of type cupy.uint8.")
    if image_cp.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    
    height, width = image_cp.shape
    block_size = (16, 16)
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])

    fill_zeros_kernel = cp.RawKernel(fill_zeros_upward_kernel_code, 'fill_zeros_upward_kernel')

    output_cp = image_cp.copy()  # or cp.empty_like(image_cp)
    fill_zeros_kernel(
        grid_size, block_size,
        (image_cp, output_cp,
         np.int32(height), np.int32(width), np.int32(margin))
    )
    return output_cp


# =============================================================================
# =============================== Local DT functions ==========================
_LOCAL_DT_KERNEL = r"""
extern "C" __global__
void local_dt_kernel(const unsigned char* __restrict__ in_map,
                     const int rows,
                     const int cols,
                     const int kernel_size,
                     float* __restrict__ out_dist)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row
    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;

    // Obstacles have distance 0
    if (in_map[idx] != 0) {
        out_dist[idx] = 0.0f;
        return;
    }

    const float SQRT2 = 1.41421356f;
    float minDist = 1e20f;

    for (int s = 1; s <= kernel_size; ++s) {
        // 1. Left
        int cx = x - s;
        if (cx >= 0) {
            int cidx = y * cols + cx;
            if (in_map[cidx] == 255) minDist = fminf(minDist, (float)s);
        }

        // 2. Right
        cx = x + s;
        if (cx < cols) {
            int cidx = y * cols + cx;
            if (in_map[cidx] == 255) minDist = fminf(minDist, (float)s);
        }

        // 3. Up
        int cy = y - s;
        if (cy >= 0) {
            int cidx = cy * cols + x;
            if (in_map[cidx] == 255) minDist = fminf(minDist, (float)s);
        }

        // 4. Up-left
        cx = x - s; cy = y - s;
        if (cx >= 0 && cy >= 0) {
            int cidx = cy * cols + cx;
            if (in_map[cidx] == 255) minDist = fminf(minDist, SQRT2 * (float)s);
        }

        // 5. Up-right
        cx = x + s; cy = y - s;
        if (cx < cols && cy >= 0) {
            int cidx = cy * cols + cx;
            if (in_map[cidx] == 255) minDist = fminf(minDist, SQRT2 * (float)s);
        }

        // 6. Up-left-mid (between Up and Up-left)
        cx = x - s/2; cy = y - s;
        if (cx >= 0 && cy >= 0) {
            int cidx = cy * cols + cx;
            if (in_map[cidx] == 255) minDist = fminf(minDist, sqrtf((s*s + (s/2)*(s/2))));
        }

        // 7. Up-right-mid (between Up and Up-right)
        cx = x + s/2; cy = y - s;
        if (cx < cols && cy >= 0) {
            int cidx = cy * cols + cx;
            if (in_map[cidx] == 255) minDist = fminf(minDist, sqrtf((s*s + (s/2)*(s/2))));
        }

        // 8. Left-up-mid (between Left and Up-left)
        cx = x - s; cy = y - s/2;
        if (cx >= 0 && cy >= 0) {
            int cidx = cy * cols + cx;
            if (in_map[cidx] == 255) minDist = fminf(minDist, sqrtf((s*s + (s/2)*(s/2))));
        }

        // 9. Right-up-mid (between Right and Up-right)
        cx = x + s; cy = y - s/2;
        if (cx < cols && cy >= 0) {
            int cidx = cy * cols + cx;
            if (in_map[cidx] == 255) minDist = fminf(minDist, sqrtf((s*s + (s/2)*(s/2))));
        }

        // Early exit for very close obstacles
        if (minDist <= 1.0f) break;
    }

    out_dist[idx] = (minDist < 1e19f) ? minDist : (float)(kernel_size + 1);
}
""";

_local_dt = cp.RawKernel(_LOCAL_DT_KERNEL, "local_dt_kernel")


def local_distance_transform(costmap_cp: cp.ndarray, kernel_size: int = 5) -> cp.ndarray:
    """
    Local distance transform using 9 directional scans (fan-shaped).

    Args:
        costmap_cp (H,W) uint8: 0=free, 255=occupied.
        kernel_size (int): maximum scan distance in pixels.

    Returns:
        (H,W) float32 CuPy array of distances.
        Obstacles: 0. Free pixels: min distance to obstacle.
    """
    if costmap_cp.ndim != 2 or costmap_cp.dtype != cp.uint8:
        raise ValueError("costmap_cp must be a 2D CuPy array with dtype=uint8.")

    rows, cols = costmap_cp.shape
    out = cp.empty((rows, cols), dtype=cp.float32)

    block = (16, 16)
    grid = ((cols + block[0] - 1) // block[0],
            (rows + block[1] - 1) // block[1])

    _local_dt(grid, block, (costmap_cp, np.int32(rows), np.int32(cols), np.int32(kernel_size), out))
    return out

# =============================================================================

# ======================================== Motion planner =======================
_LOCAL_GREEDY_KERNEL = r"""
extern "C" __global__
void greedy_trajectory_kernel(float* cost_map,
                              const int rows,
                              const int cols,
                              const int start_row,
                              const int target_row)
{
    // Each thread handles exactly one row
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Skip if outside desired band
    if (tid < start_row || tid > target_row) return;

    // Find min-cost pixel in this row
    int best_x = 0;
    float min_val = 1e20f;
    for (int c = 0; c < cols; ++c) {
        float v = cost_map[tid * cols + c];
        if (v < min_val) {
            min_val = v;
            best_x = c;
        }
    }

    // Mark it as 0 so it’s visible in the costmap output
    cost_map[tid * cols + best_x] = 0.0f;
}
"""

_greedy_traj_kernel = cp.RawKernel(_LOCAL_GREEDY_KERNEL, "greedy_trajectory_kernel")

def greedy_trajectory_costmap(cost_map_cp: cp.ndarray,
                              start_row: int,
                              target_row: int) -> cp.ndarray:
    """
    Greedy path initialization on GPU.
    Marks the min-cost pixel in each row between start_row and target_row as 0.
    """
    rows, cols = cost_map_cp.shape
    cost_map_out = cost_map_cp.copy()

    block = 128
    grid = (rows + block - 1) // block

    _greedy_traj_kernel((grid,), (block,),
                        (cost_map_out,
                         np.int32(rows),
                         np.int32(cols),
                         np.int32(start_row),
                         np.int32(target_row)))

    return cost_map_out

# ============================================================================
# ============================== Curve fitting ===================================

def fit_curve_poly_gpu(costmap_cp: cp.ndarray, degree=3, output_shape=None):
    if output_shape is None:
        output_shape = costmap_cp.shape
    H, W = output_shape
    costmap_curve_cp = cp.full((H, W), 255, dtype=cp.uint8)

    # Extract trajectory points
    traj_points = cp.argwhere(costmap_cp == 0)
    if traj_points.shape[0] < degree + 1:
        return costmap_curve_cp

    rows = traj_points[:, 0].astype(cp.float32)
    cols = traj_points[:, 1].astype(cp.float32)

    # Polynomial fit: col = f(row)
    p_coeffs = cp.polyfit(rows, cols, degree)

    # Evaluate polynomial along full row range
    row_min = cp.min(rows).item()
    row_max = cp.max(rows).item()
    row_fit = cp.arange(row_min, row_max + 1, dtype=cp.float32)

    col_fit = cp.polyval(p_coeffs, row_fit)
    col_fit = cp.clip(cp.rint(col_fit), 0, W - 1).astype(cp.int32)
    row_fit = cp.clip(cp.rint(row_fit), 0, H - 1).astype(cp.int32)

    costmap_curve_cp[row_fit, col_fit] = 0

    return costmap_curve_cp


# =========================================================================
# ============================= Trajectory Post processing ===================
# --- CUDA kernel: windowed weighted smoothing (no forcing of endpoints) ---
_TRAJ_SMOOTH_KERNEL = r"""
extern "C" __global__
void traj_smooth_kernel(const int* traj_rows,
                        const float* traj_cols,
                        int* out_cols,
                        const int n_points,
                        const int window,
                        const float start_weight,
                        const float end_weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    int half_w = window / 2;
    int start_idx = idx - half_w;
    if (start_idx < 0) start_idx = 0;
    int end_idx = idx + half_w;
    if (end_idx > n_points - 1) end_idx = n_points - 1;

    float sum_vals = 0.0f;
    float sum_weights = 0.0f;

    for (int j = start_idx; j <= end_idx; ++j) {
        float w = 1.0f;
        // give extra influence to endpoints when they are in the window
        if (j == 0) w *= start_weight;
        if (j == n_points - 1) w *= end_weight;
        sum_vals += traj_cols[j] * w;
        sum_weights += w;
    }

    float avg = 0.0f;
    if (sum_weights > 0.0f) avg = sum_vals / sum_weights;

    // clamp negative to zero
    if (avg < 0.0f) avg = 0.0f;

    out_cols[idx] = (int)(avg + 0.5f); // round to nearest int
}
""";

_traj_smooth_kernel = cp.RawKernel(_TRAJ_SMOOTH_KERNEL, "traj_smooth_kernel")


def extract_trajectory_window_smooth_gpu(costmap_cp: cp.ndarray,
                                         start_row: int,
                                         start_col: int,
                                         end_row: int,
                                         end_col: int,
                                         window: int = 20,
                                         start_weight: float = 1.0,
                                         end_weight: float = 1.0,
                                         marker_radius: int = 5,
                                         marker_value: int = 150) -> cp.ndarray:
    """
    Extract trajectory from costmap (0 pixels), smooth waypoint columns with a GPU kernel
    using a sliding window average, and return a new map with the smoothed trajectory etched.

    - Kernel does NOT force start/end columns; start/end inputs are used only for marking patches.
    """
    rows, cols = costmap_cp.shape
    traj_curve_cp = cp.full_like(costmap_cp, 255, dtype=cp.uint8)

    # Extract trajectory points (waypoints etched as 0)
    traj_rows, traj_cols = cp.nonzero(costmap_cp == 0)

    if traj_rows.size == 0:
        return traj_curve_cp

    # Sort waypoints by row (ascending)
    sort_idx = cp.argsort(traj_rows)
    traj_rows = traj_rows[sort_idx].astype(cp.int32)    # int32 for kernel
    traj_cols = traj_cols[sort_idx].astype(cp.float32)  # float32 for kernel averaging

    n_points = int(traj_rows.size)
    if n_points == 0:
        return traj_curve_cp

    # Prepare output buffer (int32)
    smoothed_cols = cp.empty((n_points,), dtype=cp.int32)

    # Launch kernel
    threads = 256
    blocks = (n_points + threads - 1) // threads

    _traj_smooth_kernel((blocks,), (threads,),
                        (traj_rows,
                         traj_cols,
                         smoothed_cols,
                         np.int32(n_points),
                         np.int32(window),
                         np.float32(start_weight),
                         np.float32(end_weight)))

    # Clamp to valid columns (0..cols-1) and ensure dtype int32
    smoothed_cols = cp.clip(smoothed_cols, 0, cols - 1).astype(cp.int32)

    # Draw smoothed trajectory into output map
    traj_curve_cp[traj_rows, smoothed_cols] = 0

    # Mark start/end patches (visualization only)
    def mark_patch(r, c):
        r_min = max(0, int(r) - marker_radius)
        r_max = min(rows, int(r) + marker_radius + 1)
        c_min = max(0, int(c) - marker_radius)
        c_max = min(cols, int(c) + marker_radius + 1)
        rr = cp.arange(r_min, r_max)[:, None]
        cc = cp.arange(c_min, c_max)[None, :]
        mask = (rr - r)**2 + (cc - c)**2 <= (marker_radius**2)
        traj_curve_cp[r_min:r_max, c_min:c_max][mask] = marker_value

    # mark using provided coordinates (these are not forced onto trajectory)
    mark_patch(start_row, start_col)
    mark_patch(end_row, end_col)

    return traj_curve_cp


# ========================================================================
# ========================= Steering Extractor ==============================

def compute_avg_steering_costmap(costmap_cp: cp.ndarray,
                                 start_row: int,
                                 lookahead_points = -5,
                                 vehicle_center_col: int = 250,
                                 vehicle_ref_row: int = 390) -> float:
    """
    Compute average steering angle in degrees for trajectory points below vehicle.

    Args:
        costmap_cp: 2D CuPy array with trajectory etched as 0
        start_row: row of vehicle front (larger index = bottom)
        lookahead_points: number of points to consider
        vehicle_center_col: column of vehicle axis (250)
        vehicle_ref_row: reference row for adjacent calculation (390)

    Returns:
        avg_angle: average steering angle in degrees
    """
    # Extract trajectory points (row, col) where path == 0
    traj_rows, traj_cols = cp.nonzero(costmap_cp == 0)
    if traj_rows.size == 0:
        return 0.0

    # Sort by row descending (so bottom rows come first)
    sort_idx = cp.argsort(-traj_rows)  
    traj_rows = traj_rows[sort_idx]
    traj_cols = traj_cols[sort_idx]

    # Take first N points that are above the vehicle front row
    mask = traj_rows > start_row-30-5 #(5 pixel look ahead) works !
    # mask = traj_rows > start_row-30-lookahead_points #(5 pixel look ahead) works !
    sel_rows = traj_rows[mask]
    # print(f"rows {sel_rows}")
    sel_cols = traj_cols[mask]
    # print(f"cols {sel_cols}")
    if sel_rows.size == 0:
        return 0.0

    # Compute geometry: opposite/adjacent for steering angle
    opposite = sel_cols - vehicle_center_col
    adjacent = start_row - sel_rows  # vertical distance (positive upwards)

    # Avoid division by zero
    adjacent = cp.where(adjacent == 0, 1e-3, adjacent)

    angles_deg = cp.degrees(cp.arctan(opposite / adjacent))
    avg_angle = float(cp.mean(angles_deg))

    return avg_angle


# ==========================================================================
# ============================ Edge Mask ===============================

edge_kernel = cp.RawKernel(r'''
extern "C" __global__
void detect_edges(const unsigned char* img, unsigned char* out, int w, int h) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;

    if (img[idx] == 0) {
        bool border = false;
        for (int dy = -1; dy <= 1 && !border; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    if (img[ny * w + nx] == 255) {
                        border = true;
                        break;
                    }
                }
            }
        }
        if (border) out[idx] = 245;
        else out[idx] = 0;
    } else {
        out[idx] = img[idx];
    }
}
''', 'detect_edges')


def detect_edges_gpu(img: cp.ndarray) -> cp.ndarray:
    h, w = img.shape
    out = cp.zeros_like(img)

    threads = (16, 16)
    blocks = ((w + threads[0] - 1) // threads[0],
              (h + threads[1] - 1) // threads[1])

    edge_kernel(blocks, threads, (img, out, w, h))
    return out

# ============================================================================
# ======================== Targeted DT 2.0 ===========================================

# Updated DT kernel with horizontal bias
dt_kernel_code = R"""
extern "C" __global__
void dt_kernel_bias(const unsigned char* costmap, unsigned char* dtmap,
                     const int H, const int W,
                     const int h_size, const int v_size,
                     const int h_bias)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= W || y >= H) return;

    int idx = y * W + x;

    if (costmap[idx] != 0) return;  // only free space

    float min_dist = 1e20f;

    int dx_min = -(h_size - h_bias);
    int dx_max = h_size + h_bias;

    for (int dy = -v_size; dy <= 0; ++dy) {
        for (int dx = dx_min; dx <= dx_max; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

            int n_idx = ny * W + nx;
            unsigned char v = costmap[n_idx];

            if (v != 245) continue; // only border pixels

            float dist = sqrtf((float)(dx*dx + dy*dy));
            if (dist < min_dist) min_dist = dist;
        }
    }

    float max_dist = sqrtf((float)(h_size*h_size + v_size*v_size));
    float normalized = fminf(min_dist / max_dist, 1.0f);
    dtmap[idx] = (unsigned char)(normalized * 255.0f);
}
"""

# Compile kernel
dt_kernel_bias = cp.RawKernel(dt_kernel_code, 'dt_kernel_bias')

def distance_transform_bias(costmap: cp.ndarray, h_size: int = 40, v_size: int = 1, h_bias: int = 0):
    """
    Distance Transform with separate horizontal and vertical kernel sizes and horizontal bias.
    
    Args:
        costmap: cp.ndarray (H, W) with 0 free, 145 border, 255 occupied
        h_size: horizontal search window radius
        v_size: vertical search window radius
        h_bias: pixels to bias horizontal search (asymmetric)
    
    Returns:
        dtmap: cp.ndarray, distance map (0 near border, 255 far)
    """
    H, W = costmap.shape
    dtmap = cp.zeros_like(costmap, dtype=cp.uint8)

    threadsperblock = (16, 16)
    blockspergrid_x = (W + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (H + threadsperblock[1] - 1) // threadsperblock[1]

    dt_kernel_bias(
        (blockspergrid_x, blockspergrid_y),
        threadsperblock,
        (costmap, dtmap, H, W, h_size, v_size, h_bias)
    )

    # Invert to match original behavior
    return 255 - dtmap

# ==========================================================================
# ============ Border Check=============================
def apply_border_halo(costmap, halo_value=120, border_value=0, kernel_size=5):
    H, W = costmap.shape
    pad = kernel_size // 2

    # Ensure dtype is uint8
    costmap = costmap.astype(cp.uint8)

    # Pad the array to handle edges
    padded = cp.pad(costmap, pad, mode='constant', constant_values=0)
    result = padded.copy()

    # Find all border pixels
    border_mask = (padded == cp.uint8(border_value))
    ys, xs = cp.nonzero(border_mask)

    # Apply halo around each border pixel
    for y, x in zip(ys, xs):
        result[y-pad:y+pad+1, x-pad:x+pad+1] = cp.uint8(halo_value)

    # Keep original border pixels as 145
    for y, x in zip(ys, xs):
        result[y, x] = cp.uint8(border_value)

    # Remove padding
    return result[pad:-pad, pad:-pad]
# ============================================================
# =========================== Throttle ===================
def estimate_throttle(dt_map_cp: cp.ndarray, etched_cp: cp.ndarray, 
                      min_throttle=0.2, max_throttle=1.0) -> float:
    """
    Estimate throttle based on DT costmap and etched trajectory.

    Args:
        dt_map_cp: cp.ndarray, distance transform costmap.
        etched_cp: cp.ndarray, costmap with trajectory etched (traj pixels = 0).
        min_throttle: minimum throttle allowed.
        max_throttle: maximum throttle allowed.
    
    Returns:
        throttle (float between min_throttle and max_throttle)
    """
    # trajectory mask (where etched map is 0)
    traj_mask = (etched_cp == 0)
    
    # DT values along trajectory
    traj_dt_vals = dt_map_cp[traj_mask]
    
    if traj_dt_vals.size == 0:
        return float(min_throttle)  # fail-safe
    
    # compute mean distance transform value along trajectory
    avg_dt = traj_dt_vals.mean()
    
    # normalize risk (smaller DT = safer, larger DT = riskier)
    # scale: if avg_dt==0 → safest, if avg_dt==dt_max → riskiest
    dt_max = dt_map_cp.max()
    risk = avg_dt / (dt_max + 1e-6)
    
    # invert risk to throttle (safe → higher throttle)
    throttle = max_throttle - (max_throttle - min_throttle) * risk
    
    # move from GPU to CPU float
    return float(throttle)
# ======================================================================
class NavStack:
    def __init__(self):
        self.start = None
        self.target = None
        self.prev_target = None
        self.x_threshold = 15
        self.fan_radius = 20
        self.prev_angle = 0

    def preprocess_costmap(self, costmap_cp):
        # print("----- No issue 0-----------------")
        if costmap_cp is None:
            return None, None, None

        # Always set self.start at the beginning
        self.start = cp.array([costmap_cp.shape[0] - 25, costmap_cp.shape[1] // 2], dtype=cp.int32)
        
        potential_targets = cp.argwhere(costmap_cp == 0)
        # print("----- No issue 1-----------------")
        if potential_targets.size == 0:
            print("Warning: Costmap has no drivable paths (value 0). Cannot find a target.")
            self.target = None # Explicitly set to None for graceful handling
            return costmap_cp , None, None
        # print("----- No issue 2-----------------")
        # --- Start of new target smoothing logic ---
        if not hasattr(self, 'prev_target') or self.prev_target is None:
            # Check if potential_targets is a 1-element array, index it accordingly
            if potential_targets.shape == (1, 2):
                self.target = potential_targets[0]
            else:
                self.target = potential_targets[0]
            print(f"Initial target set: {self.target.tolist()}")
        else:
            prev_x = self.prev_target[1]
            filtered_indices = cp.argwhere(cp.abs(potential_targets[:, 1] - prev_x) <= self.x_threshold)
            
            if filtered_indices.size > 0:
                filtered_targets = potential_targets[filtered_indices].squeeze()

                # --- FIX FOR INDEX ERROR ---
                # Check if filtered_targets is already a 1D array (a single target).
                if filtered_targets.ndim == 1:
                    self.target = filtered_targets
                else:
                    # If it's a 2D array (multiple targets), take the first one.
                    self.target = filtered_targets[0]
                # print(f"Filtered target found: {self.target.tolist()}")
            else:
                # Fallback: If no targets are within the threshold
                if potential_targets.shape == (1, 2):
                    self.target = potential_targets[0]
                else:
                    self.target = potential_targets[0]
                print(f"No targets within threshold, falling back to best available: {self.target.tolist()}")
        # print("----- No issue 3-----------------")
        self.prev_target = self.target.tolist()
        # --- End of new target smoothing logic ---
        costmap_vis = costmap_cp
        if (self.start is not None and self.start.ndim == 1 and self.start.shape[0] == 2 and
            self.target is not None and self.target.ndim == 1 and self.target.shape[0] == 2):
            costmap_vis = CostmapVisualizer.add_markers(costmap_vis, self.start, self.target, 15)
        else:
            print("Warning: Start or target position is not a valid coordinate pair. Skipping marker visualization.")
            costmap_vis = costmap_cp
        # print("----- No issue 4-----------------")
        spiked = add_zeros_right_kernel_wrapper(costmap_cp,3)
        # cv2.imshow('spiked', spiked.get())
        interpolated = add_zeros_upward_kernel_wrapper(spiked,15)
        # cv2.imshow('interpolated', interpolated.get())
        edged = detect_edges_gpu(interpolated) 
        # cv2.imshow('edged', edged.get())
        # print("----- No issue 5-----------------")
        # print(cp.sum(edged == 245))   #border detection check
        # border_check = apply_border_halo(edged)
        # cv2.imshow('border_check', border_check.get())
        target_dt = distance_transform_bias(edged)
        # cv2.imshow('target_dt', target_dt.get())
        # ======================= Gap follow =============================
        # dt_map_cp: CuPy DT costmap
        # convert to CPU for visualization
        dt_map_np = target_dt.get()
        # print(self.start)
        gap_len, steering_degree= scan_longest_gap(
            target_dt, start_row=375, start_col=250, prev_angle_deg=self.prev_angle)
        self.prev_angle = steering_degree
        # print(steering_degree)
        # print(gap_len)
        # print( (vx, vy))
        if steering_degree is not None:
            vis_img = draw_direction_arrow(dt_map_np, 375, 250, steering_degree, gap_len)

            cv2.imshow("Direction Arrow", vis_img)
        # ==================================================================
        # cv2.imshow('target_dt', target_dt.get())
        # dt_cp = local_distance_transform(interpolated, kernel_size=120)
        # # dt_cp = local_distance_transform_fan(interpolated, self.fan_offsets)
        # # # Step 2: Compute distance transform (distance from obstacle)
        # # # distance_transform_edt computes distance from zero pixels
        # # dist_map = ndi.distance_transform_edt(binary_free)

        # # # Step 3: Normalize / invert to get high values near obstacle
        # max_dist = cp.max(dt_cp)
        # dt_map = (1 - dt_cp / max_dist) * 255
        # dt_map = dt_map.astype(cp.uint8)
        # # print("----- No issue 6-----------------")
        # # cv2.imshow('dist_map',dt_map.get())


        # euclidean_costmap_cp = euclidean_costmap(target_dt)

        # try:
        #     cv2.imshow('euclidean_costmap_cp', euclidean_costmap_cp.get())
        # except Exception as e:
        #     print(f"Error displaying image: {e}")
        # # print("----- No issue 7-----------------")
        # if euclidean_costmap_cp is None:
        #     print("Euclidean costmap is None, returning from preprocess_costmap.")
        #     return None, None

        # if euclidean_costmap_cp.size == 0:
        #     print("Warning: No suitable targets found after costmap processing (cost < 50).")
        #     return None, None
        # else:
        #     # This line is now safe because the self.target variable is guaranteed to be a 1D array
        #     targ = [self.target[0], self.target[1]]
        # # print("----- No issue 8-----------------")
        # # print(f"target = {self.target} start = {self.start}")
        # target_row = self.target[0].item()
        # # print(type(target_row))  # <class 'int'>
        
        # euclidean_costmap_targ_cp = euclidean_costmap_from_target(euclidean_costmap_cp, target_cp=targ)
        # cv2.imshow('euclidean_costmap_targ_cp', euclidean_costmap_targ_cp.get())
        # rows, cols = euclidean_costmap_targ_cp.shape
        # start_r = target_row + 5
        # end_r   = 365  # your fixed value
        # # print("----- No issue 9-----------------")
        # # Only run if both within valid range and start < end
        # if 0 <= start_r < rows and 0 <= end_r < rows and start_r < end_r:
        #     path_etched_costmap = plan_gap_follow_gpu(
        #         euclidean_costmap_targ_cp,
        #         start_r,
        #         end_r,
        #         smooth_weight=100.0,
        #         iterations=3,
        #         search_radius=5
        #     )
        #     cv2.imshow('path_etched_costmap', path_etched_costmap.get())
        # else:
        #     return None , None
        # # try:
        # #     cv2.imshow('path_etched_costmap', path_etched_costmap.get())
        # # except Exception as e:
        # #     print(f"Error displaying image: {e}")
        # if path_etched_costmap is None:
        #     return None, None
        # # print("----- No issue 10-----------------")
        # traj_window_cp = extract_trajectory_window_smooth_gpu(path_etched_costmap,390,250,self.target[0].item(),self.target[1].item())
        # cv2.imshow('trajectory_window', traj_window_cp.get())
        # if traj_window_cp is None:
        #     return None, None
        # # print("----- No issue 11-----------------")
        # # border_check = apply_border_halo(edged)
        # # cv2.imshow('border_check', border_check.get())
        # steering_angle = compute_avg_steering_costmap(traj_window_cp,390,250)
        # if steering_angle is None:
        #     return costmap_vis, None  # safe fallback
        # # print(steering_angle)
        steering = (max(-30, min(30, steering_degree)))/30
        throttle = (max(0, min(300, gap_len)))/300
        # print(throttle)
        # steering = 1
        #throttle Calculation
        # throttle = estimate_throttle(euclidean_costmap_targ_cp, path_etched_costmap)
        # print("Throttle command:", throttle)

        return costmap_vis , steering , throttle   


class CostmapVisualizer:
    """
    A utility class to add visual markers to a GPU-based costmap.

    This is useful for debugging and verifying the start and goal positions
    for the motion planner. All operations are performed on the GPU.
    """
    @staticmethod
    def add_markers(costmap_cp, start_pos_cp, goal_pos_cp, marker_size=5):
        """
        Adds square markers for the start and goal positions to the costmap.

        Args:
            costmap_cp (cupy.ndarray): The input costmap.
            start_pos_cp (cupy.ndarray): A 1D CuPy array [y, x] for the start position.
            goal_pos_cp (cupy.ndarray): A 1D CuPy array [y, x] for the goal position.
            marker_size (int): The size of the square marker (e.g., 5 means a 5x5 square).

        Returns:
            cupy.ndarray: The costmap with the markers drawn on it.
        """
        # Ensure the marker size is odd so it can be centered
        if marker_size % 2 == 0:
            marker_size += 1
        half_size = marker_size // 2

        # Get the dimensions of the costmap
        rows, cols = costmap_cp.shape
        if rows == 0:
            return costmap_cp

        def _draw_marker(pos_cp, value):
            """
            Helper function to draw a single marker on the costmap.
            """
            y_center, x_center = pos_cp
            
            # Define the bounding box for the marker, clamping to grid boundaries
            y_min = cp.maximum(0, y_center - half_size)
            y_max = cp.minimum(rows, y_center + half_size + 1)
            x_min = cp.maximum(0, x_center - half_size)
            x_max = cp.minimum(cols, x_center + half_size + 1)
            
            # Use CuPy slicing to set the pixels to the marker value
            costmap_cp[y_min:y_max, x_min:x_max] = value

        # Draw the start marker (e.g., value 100)
        _draw_marker(start_pos_cp, 0)

        # Draw the goal marker (e.g., value 150)
        _draw_marker(goal_pos_cp, 0)

        return costmap_cp
