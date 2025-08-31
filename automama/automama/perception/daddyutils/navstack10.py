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
#==========================Min Path sorting Kernel and function=====================
# 1) Initialize greedy trajectory: argmin per row
_INIT_GREEDY_KERNEL = r"""
extern "C" __global__
void init_greedy_traj(const unsigned char* __restrict__ costmap,
                      const int rows,
                      const int cols,
                      const int start_row,
                      const int end_row,
                      int* __restrict__ traj_x)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < start_row || r > end_row || r >= rows) return;

    int best_c = 0;
    int best_v = 255;
    const int base = r * cols;

    for (int c = 0; c < cols; ++c) {
        int v = (int)costmap[base + c];
        if (v < best_v) { best_v = v; best_c = c; }
    }
    traj_x[r] = best_c;
}
""";

# 2) One smoothing step (Jacobi update)
#    total_cost = pos_cost + smooth_weight * ((cand - prev)^2 + (cand - next)^2)
#    Search cand in [x-search_radius, x+search_radius]
_SMOOTH_STEP_KERNEL = r"""
extern "C" __global__
void smooth_step(const unsigned char* __restrict__ costmap,
                 const int rows,
                 const int cols,
                 const int start_row,
                 const int end_row,
                 const float smooth_weight,
                 const int search_radius,
                 const int* __restrict__ traj_in,
                 int* __restrict__ traj_out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < start_row || r > end_row || r >= rows) return;

    // Pin endpoints
    if (r == start_row || r == end_row) {
        traj_out[r] = traj_in[r];
        return;
    }

    int x_prev = traj_in[r - 1];
    int x_curr = traj_in[r];
    int x_next = traj_in[r + 1];

    int base = r * cols;

    float best_cost = 1e30f;
    int   best_x = x_curr;

    int cmin = x_curr - search_radius;
    int cmax = x_curr + search_radius;
    if (cmin < 0) cmin = 0;
    if (cmax >= cols) cmax = cols - 1;

    for (int cand = cmin; cand <= cmax; ++cand) {
        float pos_cost = (float)costmap[base + cand];
        float dxp = (float)(cand - x_prev);
        float dxn = (float)(cand - x_next);
        float smooth_cost = smooth_weight * (dxp*dxp + dxn*dxn);
        float total = pos_cost + smooth_cost;

        if (total < best_cost) {
            best_cost = total;
            best_x = cand;
        }
    }

    traj_out[r] = best_x;
}
""";

# 3) Paint final trajectory into the costmap (set pixels to 0 for visualization)
_PAINT_TRAJ_KERNEL = r"""
extern "C" __global__
void paint_traj(unsigned char* __restrict__ costmap,
                const int rows,
                const int cols,
                const int start_row,
                const int end_row,
                const int* __restrict__ traj_x)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < start_row || r > end_row || r >= rows) return;

    int c = traj_x[r];
    if (c >= 0 && c < cols) {
        costmap[r * cols + c] = (unsigned char)0;
    }
}
""";

_init_greedy = cp.RawKernel(_INIT_GREEDY_KERNEL, "init_greedy_traj")
_smooth_step = cp.RawKernel(_SMOOTH_STEP_KERNEL, "smooth_step")
_paint_traj  = cp.RawKernel(_PAINT_TRAJ_KERNEL,  "paint_traj")

def smooth_greedy_pipeline(costmap_cp: cp.ndarray,
                           start_row: int,
                           end_row: int,
                           smooth_weight: float = 20.0,
                           iterations: int = 5,
                           search_radius: int = 1) -> cp.ndarray:
    """
    Inputs:
        costmap_cp   : (H,W) uint8 CuPy array. 0=free, 255=occupied (or any cost 0..255).
        start_row    : inclusive
        end_row      : inclusive
        smooth_weight: tradeoff vs positional cost (try 5..20)
        iterations   : smoothing passes
        search_radius: columns to each side to consider per iteration

    Output:
        modified costmap (uint8) where optimized trajectory pixels are set to 0.
    """
    if costmap_cp.ndim != 2 or costmap_cp.dtype != cp.uint8:
        raise ValueError("costmap_cp must be (H,W) with dtype=uint8")
    rows, cols = costmap_cp.shape
    
    if start_row < 0 or end_row >= rows or start_row > end_row:
        raise ValueError("Invalid start/end row")

    # Copy, we will draw on it
    out_map = costmap_cp.copy()

    # Allocate trajectory buffers (ping-pong)
    traj_a = cp.full((rows,), -1, dtype=cp.int32)
    traj_b = cp.full((rows,), -1, dtype=cp.int32)

    block = 128
    grid  = (rows + block - 1) // block

    # Step 1: initialize greedy per-row minima
    _init_greedy((grid,), (block,), (
        out_map,
        np.int32(rows), np.int32(cols),
        np.int32(start_row), np.int32(end_row),
        traj_a
    ))

    # Step 2: smoothing iterations (Jacobi style)
    for it in range(iterations):
        _smooth_step((grid,), (block,), (
            out_map,
            np.int32(rows), np.int32(cols),
            np.int32(start_row), np.int32(end_row),
            np.float32(smooth_weight),
            np.int32(search_radius),
            traj_a,   # in
            traj_b    # out
        ))
        # swap
        traj_a, traj_b = traj_b, traj_a

    # Step 3: paint the final trajectory on the map (set to 0 for visualization)
    _paint_traj((grid,), (block,), (
        out_map,
        np.int32(rows), np.int32(cols),
        np.int32(start_row), np.int32(end_row),
        traj_a
    ))

    return out_map

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

# Kernel code
dt_kernel_code = r'''
extern "C" __global__
void dt_kernel(const unsigned char* costmap, unsigned char* dtmap,
               const int H, const int W, const int kernel_radius)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= W || y >= H) return;

    int idx = y * W + x;

    // Only process free space pixels
    if (costmap[idx] != 0) return;

    float min_dist = 1e20f;  // Initialize a very large distance

    // Search only upper half of kernel
    for (int dy = -kernel_radius; dy <= 0; ++dy) {
        for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx < 0 || nx >= W || ny < 0) continue;

            int n_idx = ny * W + nx;
            unsigned char v = costmap[n_idx];

            // Only consider border pixels (145)
            if (v != 245) continue;

            // Compute distance to this border pixel
            float dist = sqrtf((float)(dx*dx + dy*dy));

            if (dist < min_dist) min_dist = dist;
        }
    }

    // Smooth gradient: map distance to [0, 255] inversely
    // You can scale by kernel_radius to control smoothness
    // map distance to DT: 0 at obstacle, 255 at farthest free space
    float normalized = fminf(min_dist / (float)kernel_radius, 1.0f);
    dtmap[idx] = (unsigned char)((normalized) * 255.0f);

}
'''

# Compile kernel
dt_kernel = cp.RawKernel(dt_kernel_code, 'dt_kernel')

def distance_transform(costmap, kernel_radius=5):
    """
    costmap: cp.ndarray (H, W) with 0 free, 145 border, 255 occupied
    kernel_radius: search window size
    """
    H, W = costmap.shape
    dtmap = cp.zeros_like(costmap, dtype=cp.uint8)

    threadsperblock = (16, 16)
    blockspergrid_x = (W + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (H + threadsperblock[1] - 1) // threadsperblock[1]

    dt_kernel(
        (blockspergrid_x, blockspergrid_y),
        threadsperblock,
        (costmap, dtmap, H, W, kernel_radius)
    )

    return (255-dtmap)

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
# ==============================
# 1) CUDA kernel: waypoint means
# ==============================
# - Operates ONLY on waypoint pixels (costmap==0).
# - Computes mean over a KxK neighborhood, ignoring 0-valued pixels.
# - If ANY 255 is present in the window, writes 255 immediately.
# - Writes result into meanmap at the waypoint pixel location.
_TRAJECTORY_MEAN_KERNEL = r"""
extern "C" __global__
void trajectory_mean_kernel(const unsigned char* __restrict__ costmap,
                            unsigned char* __restrict__ meanmap,
                            const int H,
                            const int W,
                            const int kernel_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = H * W;
    if (idx >= total) return;

    // process ONLY waypoint pixels
    if (costmap[idx] != 0) return;

    int y = idx / W;
    int x = idx % W;

    // define symmetric window of size kernel_size even/odd:
    // for even K, we cover exactly K cells: lo = -K/2, hi = K - K/2 - 1
    int lo = - (kernel_size / 2);
    int hi =  (kernel_size - (kernel_size / 2) - 1);

    int sumv = 0;
    int cnt  = 0;

    // scan window
    for (int dy = lo; dy <= hi; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= H) continue;

        for (int dx = lo; dx <= hi; ++dx) {
            int nx = x + dx;
            if (nx < 0 || nx >= W) continue;

            unsigned char v = costmap[ny * W + nx];

            // if obstacle found -> immediate 255 at this waypoint
            if (v == 255) {
                meanmap[idx] = (unsigned char)255;
                return;
            }

            // ignore zeros when computing mean
            if (v != 0) {
                sumv += (int)v;
                cnt  += 1;
            }
        }
    }

    // write mean (uint8). If all were zero, keep 0.
    if (cnt > 0) {
        int meanv = sumv / cnt;
        if (meanv > 255) meanv = 255; // safety
        meanmap[idx] = (unsigned char)meanv;
    } else {
        meanmap[idx] = (unsigned char)255;
    }
}
""";

trajectory_mean_kernel = cp.RawKernel(_TRAJECTORY_MEAN_KERNEL, "trajectory_mean_kernel")


def compute_waypoint_meanmap(costmap_cp: cp.ndarray, kernel_size: int = 4) -> cp.ndarray:
    """
    Compute meanmap for waypoint pixels (==0) using a KxK window.
    - Ignores 0s in window, returns 255 immediately if any 255 present.
    - Writes means at waypoint locations; non-waypoint pixels stay 0.

    Args:
        costmap_cp: (H,W) uint8 CuPy array
        kernel_size: neighborhood size (even like 4 or odd like 5)

    Returns:
        meanmap_cp: (H,W) uint8 CuPy array
    """
    if costmap_cp.ndim != 2 or costmap_cp.dtype != cp.uint8:
        raise ValueError("costmap_cp must be (H,W) uint8")

    H, W = costmap_cp.shape
    meanmap = cp.zeros_like(costmap_cp, dtype=cp.uint8)

    threads = 256
    blocks  = (H * W + threads - 1) // threads

    trajectory_mean_kernel((blocks,), (threads,),
                           (costmap_cp, meanmap, np.int32(H), np.int32(W), np.int32(kernel_size)))
    return meanmap


# ===============================================
# 2) Scan trajectory upward & measure valid chain
# ===============================================
def measure_trajectory_chain_length(costmap_cp: cp.ndarray,
                                    meanmap_cp: cp.ndarray,
                                    start_row: int = 390,
                                    end_row:   int = 100,
                                    vehicle_center_col: int = 250,
                                    stop_threshold: int = 150,
                                    count_ceiling: int = 200) -> int:
    """
    Walk from start_row down to end_row (i.e., upward in image),
    following waypoint pixels (costmap==0). For each row, choose the
    waypoint column closest to the previous column (start from vehicle_center_col).
    Count length until:
      - first mean > stop_threshold (stop and do NOT count that pixel)
      - or no waypoint found in a row (break)
    Only count pixels whose mean < count_ceiling.

    Args:
        costmap_cp: (H,W) uint8 with waypoints etched as 0
        meanmap_cp: (H,W) uint8 with per-waypoint means computed above
        start_row: bottom/vehicle row to start (e.g., 390)
        end_row: top bound (e.g., 100) inclusive
        vehicle_center_col: starting reference column (e.g., 250)
        stop_threshold: first mean > this -> stop
        count_ceiling: only count means strictly below this (e.g., 200)

    Returns:
        length (int): number of continuous rows counted
    """
    H, W = costmap_cp.shape
    if not (0 <= end_row <= start_row < H):
        raise ValueError("Invalid start_row/end_row")

    # Precompute waypoint mask for rows of interest
    waypoint_mask = (costmap_cp == 0)

    length = 0
    prev_c = vehicle_center_col

    # Iterate rows: start_row, start_row-1, ..., end_row
    for r in range(start_row, end_row - 1, -1):
        row_mask = waypoint_mask[r]
        if not row_mask.any():
            break  # gap -> stop

        cols = cp.where(row_mask)[0]
        if cols.size == 0:
            break

        # Choose closest waypoint to previous column
        # (works well if there are multiple waypoints in a row)
        diffs = cp.abs(cols - prev_c)
        best_idx = int(cp.argmin(diffs).get())
        c = int(cols[best_idx].get())

        m = int(meanmap_cp[r, c].get())

        # Stop rule: first mean > stop_threshold -> stop (do not count)
        if m > stop_threshold:
            break

        # Count rule: only count means < count_ceiling
        if m < count_ceiling:
            length += 1
        else:
            # mean is within [stop_threshold .. count_ceiling), you can decide:
            # here we do not stop (since stop is only > stop_threshold),
            # but we also do not count. If you prefer to stop here, uncomment:
            # break
            pass

        prev_c = c

    return length
# ======================================================================
class NavStack:
    def __init__(self):
        self.start = None
        self.target = None
        self.prev_target = None
        self.x_threshold = 15
        self.fan_radius = 20

    def preprocess_costmap(self, costmap_cp):
        # print("----- No issue 0-----------------")
        if costmap_cp is None:
            return None, None

        # Always set self.start at the beginning
        self.start = cp.array([costmap_cp.shape[0] - 25, costmap_cp.shape[1] // 2], dtype=cp.int32)
        
        potential_targets = cp.argwhere(costmap_cp == 0)
        # print("----- No issue 1-----------------")
        if potential_targets.size == 0:
            print("Warning: Costmap has no drivable paths (value 0). Cannot find a target.")
            self.target = None # Explicitly set to None for graceful handling
            return costmap_cp , None
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
        target_dt = distance_transform(edged,50)
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


        euclidean_costmap_cp = euclidean_costmap(target_dt)

        # try:
        #     cv2.imshow('euclidean_costmap_cp', euclidean_costmap_cp.get())
        # except Exception as e:
        #     print(f"Error displaying image: {e}")
        # print("----- No issue 7-----------------")
        if euclidean_costmap_cp is None:
            print("Euclidean costmap is None, returning from preprocess_costmap.")
            return None, None

        if euclidean_costmap_cp.size == 0:
            print("Warning: No suitable targets found after costmap processing (cost < 50).")
            return None, None
        else:
            # This line is now safe because the self.target variable is guaranteed to be a 1D array
            targ = [self.target[0], self.target[1]]
        # print("----- No issue 8-----------------")
        # print(f"target = {self.target} start = {self.start}")
        target_row = self.target[0].item()
        # print(type(target_row))  # <class 'int'>
        
        euclidean_costmap_targ_cp = euclidean_costmap_from_target(euclidean_costmap_cp, target_cp=targ)

        rows, cols = euclidean_costmap_targ_cp.shape
        start_r = target_row + 5
        end_r   = 365  # your fixed value
        # print("----- No issue 9-----------------")
        # Only run if both within valid range and start < end
        if 0 <= start_r < rows and 0 <= end_r < rows and start_r < end_r:
            path_etched_costmap = smooth_greedy_pipeline(
                euclidean_costmap_targ_cp,
                start_r,
                end_r,
                smooth_weight=100.0,
                iterations=3,
                search_radius=5
            )
            cv2.imshow('path_etched_costmap', path_etched_costmap.get())
        else:
            return None , None
        # try:
        #     cv2.imshow('path_etched_costmap', path_etched_costmap.get())
        # except Exception as e:
        #     print(f"Error displaying image: {e}")
        if path_etched_costmap is None:
            return None, None
        # print("----- No issue 10-----------------")
        traj_window_cp = extract_trajectory_window_smooth_gpu(path_etched_costmap,390,250,self.target[0].item(),self.target[1].item())
        cv2.imshow('trajectory_window', traj_window_cp.get())
        if traj_window_cp is None:
            return None, None
        # print("----- No issue 11-----------------")
        # border_check = apply_border_halo(edged)
        # cv2.imshow('border_check', border_check.get())
        steering_angle = compute_avg_steering_costmap(traj_window_cp,390,250)
        if steering_angle is None:
            return costmap_vis, None  # safe fallback
        # print(steering_angle)
        steering = (max(-30, min(30, steering_angle)))/30
        
        # steering = 1
        #throttle Calculation
        # # 1) Compute meanmap over waypoints with 4x4 window
        # meanmap = compute_waypoint_meanmap(path_etched_costmap, kernel_size=4)

        # # 2) Measure chain length from vehicle row 390 up to row 300
        # length = measure_trajectory_chain_length(path_etched_costmap, meanmap,
        #                                         start_row=390,
        #                                         end_row=300,
        #                                         vehicle_center_col=250,
        #                                         stop_threshold=150,
        #                                         count_ceiling=200)

        # print("Valid trajectory chain length:", length)

        return costmap_vis , steering    


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
        _draw_marker(start_pos_cp, 100)

        # Draw the goal marker (e.g., value 150)
        _draw_marker(goal_pos_cp, 150)

        return costmap_cp
