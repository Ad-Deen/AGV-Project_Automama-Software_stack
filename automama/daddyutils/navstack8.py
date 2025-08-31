import cupy as cp
import numpy as np
from queue import PriorityQueue
import cv2
import cupyx.scipy.ndimage as ndi
from cupyx.scipy import interpolate
# ==============================Gradient inflation kernel and function ===============================
# --- 1. Define the CUDA kernel for selective blurring ---
# The kernel takes an input array, a boolean mask, and an output array.
# It only applies the blur effect where the mask is True.
Blur_KERNEL_SOURCE = r"""
extern "C" __global__
void blur_pixels_kernel(const unsigned char* input,
                        unsigned char* output,
                        int height,
                        int width,
                        int kernel_size,
                        const float* gaussian_weights) {

    // Get the unique 2D index for this thread.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image bounds.
    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    int kernel_radius = kernel_size / 2;

    // Only apply the blur if the pixel's value is 0.
    if (input[idx] == 0) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        // Iterate over the custom-sized kernel neighborhood.
        for (int j = -kernel_radius; j <= kernel_radius; ++j) {
            for (int i = -kernel_radius; i <= kernel_radius; ++i) {
                int neighbor_x = x + i;
                int neighbor_y = y + j;
                int kernel_idx = (j + kernel_radius) * kernel_size + (i + kernel_radius);
                float weight = gaussian_weights[kernel_idx];

                // Check for bounds to avoid out-of-bounds memory access.
                if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                    int neighbor_idx = neighbor_y * width + neighbor_x;
                    sum += input[neighbor_idx] * weight;
                    weight_sum += weight;
                }
            }
        }
        
        // Calculate the weighted average and set the output pixel value.
        // We use weight_sum to handle pixels at the image borders gracefully.
        if (weight_sum > 0) {
            output[idx] = (unsigned char)(sum / weight_sum);
        } else {
            output[idx] = input[idx]; // Fallback if no neighbors were found
        }
    } else {
        // If the pixel is not 0, just copy the original value to the output.
        output[idx] = input[idx];
    }
}
"""

def blur_zero_pixels(image_cp, kernel_size=3, sigma=1.0):
    """
    Applies a Gaussian blur only to pixels with a value of 0.

    Args:
        image_cp (cupy.ndarray): The input 2D CuPy array (e.g., a grayscale image).
        kernel_size (int): The size of the blur kernel (e.g., 3 for 3x3, 5 for 5x5).
                           Must be an odd number.
        sigma (float): The standard deviation of the Gaussian function,
                       controlling the blur strength.

    Returns:
        cupy.ndarray: The output image with selective blurring applied.
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        print("Warning: Kernel size must be an odd number. Using next odd number.")
        kernel_size += 1

    # Generate Gaussian kernel weights on the CPU.
    kernel_radius = kernel_size // 2
    x, y = np.mgrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    gaussian_kernel /= np.sum(gaussian_kernel) # Normalize the weights
    
    # Flatten the kernel and transfer to the GPU.
    gaussian_weights_cp = cp.asarray(gaussian_kernel.flatten(), dtype=cp.float32)

    height, width = image_cp.shape
    
    # Create the output array on the GPU.
    output_cp = cp.empty_like(image_cp)

    # Get the kernel from the source code.
    blur_kernel = cp.RawKernel(Blur_KERNEL_SOURCE, 'blur_pixels_kernel')

    # Define the kernel launch configuration.
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel with the new parameters.
    blur_kernel(blocks_per_grid, threads_per_block,
                (image_cp, output_cp, height, width, kernel_size, gaussian_weights_cp))
    
    return output_cp

# ==================================================================================================
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
def create_disk_kernel(radius):
    """Create a 2D disk-shaped structuring element."""
    L = np.arange(-radius, radius+1)
    X, Y = np.meshgrid(L, L)
    disk = (X**2 + Y**2) <= radius**2
    return cp.array(disk, dtype=cp.uint8)


def precompute_semi_circle_offsets(radius: int, num_points: int = 36):
    """
    Generates (dy, dx) offsets forming a semi-circle above the pixel.
    Angles go from pi (left) to 0 (right), for upper half.

    Returns:
        offsets: (N,2) int array of [dy, dx]
    """
    angles = np.linspace(np.pi, 0, num_points)
    offsets = []
    for theta in angles:
        dy = -int(round(radius * np.sin(theta)))  # negative because upward
        dx = int(round(radius * np.cos(theta)))
        offsets.append([dy, dx])
    return np.array(offsets, dtype=np.int32)
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
# =================================ARC DT ===============================
_LOCAL_ARC_DT_KERNEL = r"""
extern "C" __global__
void local_dt_arc_kernel_offsets(const unsigned char* __restrict__ in_map,
                             const int rows,
                             const int cols,
                             const int num_offsets,
                             const int* __restrict__ offsets,  // [dy, dx] pairs
                             float* __restrict__ out_dist)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;

    if (in_map[idx] != 0) {
        out_dist[idx] = 0.0f;
        return;
    }

    float minDist = 1e20f;

    for (int i = 0; i < num_offsets; ++i) {
        int dy = offsets[2*i + 0];
        int dx = offsets[2*i + 1];

        int ny = y + dy;
        int nx = x + dx;

        if (ny >= 0 && ny < rows && nx >= 0 && nx < cols) {
            int nidx = ny * cols + nx;
            if (in_map[nidx] == 255) {
                float dist = sqrtf((float)(dx*dx + dy*dy));
                if (dist < minDist) minDist = dist;
            }
        }
    }

    out_dist[idx] = (minDist < 1e19f) ? minDist : 1e20f;
}
"""
_local_dt_offsets = cp.RawKernel(_LOCAL_ARC_DT_KERNEL, "local_dt_arc_kernel_offsets")
def local_distance_transform_fan(costmap_cp: cp.ndarray, offsets_cp: cp.ndarray) -> cp.ndarray:
    """
    costmap_cp: 2D uint8 CuPy array
    offsets_cp: (N,2) CuPy array of precomputed offsets
    """
    rows, cols = costmap_cp.shape
    out = cp.empty((rows, cols), dtype=cp.float32)

    block = (16, 16)
    grid = ((cols + block[0]-1)//block[0], (rows + block[1]-1)//block[1])

    _local_dt_offsets(grid, block, (
        costmap_cp,
        np.int32(rows),
        np.int32(cols),
        np.int32(offsets_cp.shape[0]),
        offsets_cp,
        out
    ))
    return out


# ===========================================================================
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
class NavStack:
    def __init__(self):
        self.start = None
        self.target = None
        self.prev_target = None
        self.x_threshold = 15
        self.fan_radius = 20
        # self.structure = create_disk_kernel(10)
        # cp.asarray(offsets.flatten(), dtype=cp.int32)
        self.fan_offsets = cp.asarray(precompute_semi_circle_offsets(radius=self.fan_radius).flatten(), dtype=cp.int32)
    def preprocess_costmap(self, costmap_cp):
            if costmap_cp is None:
                return None

            # Always set self.start at the beginning
            self.start = cp.array([costmap_cp.shape[0] - 25, costmap_cp.shape[1] // 2], dtype=cp.int32)
            
            potential_targets = cp.argwhere(costmap_cp == 0)

            if potential_targets.size == 0:
                print("Warning: Costmap has no drivable paths (value 0). Cannot find a target.")
                self.target = None # Explicitly set to None for graceful handling
                return costmap_cp 
            
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

            self.prev_target = self.target.tolist()
            # --- End of new target smoothing logic ---
            costmap_vis = costmap_cp
            if (self.start is not None and self.start.ndim == 1 and self.start.shape[0] == 2 and
                self.target is not None and self.target.ndim == 1 and self.target.shape[0] == 2):
                costmap_vis = CostmapVisualizer.add_markers(costmap_vis, self.start, self.target, 15)
            else:
                print("Warning: Start or target position is not a valid coordinate pair. Skipping marker visualization.")
                costmap_vis = costmap_cp
            
            spiked = add_zeros_right_kernel_wrapper(costmap_cp,3)
            # cv2.imshow('spiked', spiked.get())
            interpolated = add_zeros_upward_kernel_wrapper(spiked,15)
            cv2.imshow('interpolated', interpolated.get())

            dt_cp = local_distance_transform(interpolated, kernel_size=120)
            # dt_cp = local_distance_transform_fan(interpolated, self.fan_offsets)
            # # Step 2: Compute distance transform (distance from obstacle)
            # # distance_transform_edt computes distance from zero pixels
            # dist_map = ndi.distance_transform_edt(binary_free)

            # # Step 3: Normalize / invert to get high values near obstacle
            max_dist = cp.max(dt_cp)
            dt_map = (1 - dt_cp / max_dist) * 255
            dt_map = dt_map.astype(cp.uint8)
     
            # cv2.imshow('dist_map',dt_map.get())
            cv2.imshow('dist_map',dt_map.get())

            euclidean_costmap_cp = euclidean_costmap(dt_map)

            # try:
            #     cv2.imshow('euclidean_costmap_cp', euclidean_costmap_cp.get())
            # except Exception as e:
            #     print(f"Error displaying image: {e}")

            if euclidean_costmap_cp is None:
                print("Euclidean costmap is None, returning from preprocess_costmap.")
                return None

            if euclidean_costmap_cp.size == 0:
                print("Warning: No suitable targets found after costmap processing (cost < 50).")
                return None
            else:
                # This line is now safe because the self.target variable is guaranteed to be a 1D array
                targ = [self.target[0], self.target[1]]

            # print(f"target = {self.target} start = {self.start}")
            target_row = self.target[0].item()
            # print(type(target_row))  # <class 'int'>
            euclidean_costmap_targ_cp = euclidean_costmap_from_target(euclidean_costmap_cp, target_cp=targ)
            # try:
            #     cv2.imshow('euclidean_costmap_targ_cp', euclidean_costmap_targ_cp.get())
            # except Exception as e:
            #     print(f"Error displaying image: {e}")
            # print(f"Max value in costmap before thresholding: {euclidean_costmap_targ_cp.max()}")
            # print(f"Min value in costmap before thresholding: {euclidean_costmap_targ_cp.min()}")

            # # Call the GPU function and capture the returned array.
            # trajectory = smooth_trajectory_costmap(euclidean_costmap_targ_cp,250,target_row,smooth_weight=1,iterations=5)
            # trajectory=greedy_trajectory_costmap(euclidean_costmap_targ_cp,)
            path_etched_costmap = smooth_greedy_pipeline(euclidean_costmap_targ_cp,target_row+5,370,smooth_weight=5.0,iterations=5,search_radius=2)
            try:
                cv2.imshow('path_etched_costmap', path_etched_costmap.get())
            except Exception as e:
                print(f"Error displaying image: {e}")
            # smoothed_curve_map_gpu = fit_curve_poly_gpu(path_etched_costmap, degree=5)
            # try:
            #     cv2.imshow('smoothed_curve_map_gpu', smoothed_curve_map_gpu.get())
            # except Exception as e:
            #     print(f"Error displaying image: {e}")
            return costmap_vis























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
