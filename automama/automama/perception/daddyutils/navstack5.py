import cupy as cp
import numpy as np
from queue import PriorityQueue
import cv2
import cupyx.scipy.ndimage as ndi

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
            int new_cost = costmap[index] + (int)euclidean_cost * 0.5; //scaling factor = 1.0
            
            // Ensure the new cost does not exceed the maximum value of 255
            costmap[index] = (unsigned char)((new_cost < 255) ? new_cost : 255);
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
            int new_cost = costmap[index] + (int)(distance) *0.5; 
            
            // Ensure the new cost does not exceed the maximum value of 255
            costmap[index] = (unsigned char)((new_cost < 255) ? new_cost : 255);
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
# CORRECTED: The kernel now uses `unsigned char*` to match the data type of the costmap.
# The comparison and assignment values are now also simple integers.
MIN_COST_REDUCTION_KERNEL = r"""
extern "C" __global__ void reduce_min_cost_in_row_kernel(unsigned char* costmap, int rows, int cols, int pixels_to_reduce) {
    // Each thread is responsible for one row.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Use a small, fixed-size array to store the indices and costs of the lowest-cost pixels.
        // We'll use a simple bubble sort-like approach to keep it sorted.
        // This is efficient enough for a small number of pixels_to_reduce.
        unsigned int min_costs[10]; // 10 is the max for pixels_to_reduce
        int min_indices[10];
        
        // Initialize with a high cost and an invalid index.
        for (int i = 0; i < pixels_to_reduce; ++i) {
            min_costs[i] = 255;
            min_indices[i] = -1;
        }

        // Iterate through the columns of the current row to find the minimum costs.
        for (int col = 0; col < cols; ++col) {
            int index = row * cols + col;
            unsigned char current_cost = costmap[index];

            // Only consider pixels with costs less than the threshold (230) and not an obstacle.
            if (current_cost < 250) {
                // Find the correct position to insert the new cost into our sorted array.
                for (int i = 0; i < pixels_to_reduce; ++i) {
                    if (current_cost < min_costs[i]) {
                        // Shift existing elements to make room for the new one.
                        for (int j = pixels_to_reduce - 1; j > i; --j) {
                            min_costs[j] = min_costs[j - 1];
                            min_indices[j] = min_indices[j - 1];
                        }
                        // Insert the new element.
                        min_costs[i] = current_cost;
                        min_indices[i] = index;
                        break;
                    }
                }
            }
        }
        
        // Now, reduce the cost for the selected minimum-cost pixels by 50%.
        for (int i = 0; i < pixels_to_reduce; ++i) {
            if (min_indices[i] != -1) {
                unsigned char current_cost = costmap[min_indices[i]];
                costmap[min_indices[i]] = (unsigned char)((float)current_cost * 0.0f);
            }
        }
    }
}
"""

def reduce_min_cost_on_gpu(costmap_cp, pixels_to_reduce=5):
    """
    Finds the lowest cost pixels in each row and reduces their cost by 50%.
    This function launches a 1D kernel where each thread processes a single row.

    Args:
        costmap_cp (cupy.ndarray): The input costmap as a CuPy array.
        pixels_to_reduce (int): The number of minimum-cost pixels to reduce per row.

    Returns:
        cupy.ndarray: The modified costmap.
    """
    if costmap_cp.ndim != 2:
        print("Error: Input costmap must be a 2D CuPy array.")
        return None

    rows, cols = costmap_cp.shape
    
    # Load the new kernel.
    reduce_kernel = cp.RawKernel(MIN_COST_REDUCTION_KERNEL, 'reduce_min_cost_in_row_kernel')

    # Define the block and grid dimensions for a 1D kernel launch (one thread per row).
    block_dim = (128,)  
    grid_dim = ((rows + block_dim[0] - 1) // block_dim[0],)
    
    # Launch the kernel.
    try:
        reduce_kernel(
            grid_dim,
            block_dim,
            (costmap_cp, rows, cols, pixels_to_reduce)
        )
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"CUDA Runtime Error: {e}")
        return None

    return costmap_cp

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




class NavStack:
    def __init__(self):
        self.start = None
        self.target = None
        self.prev_target = None
        self.x_threshold = 15

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
            # cv2.imshow('interpolated', interpolated.get())


            # rows_with_zeros_mask = cp.any(costmap_cp == 0, axis=1)
            # resized_costmap_cp = costmap_cp[rows_with_zeros_mask]


            gradient_costmap = blur_zero_pixels(interpolated,67,30)
            # try:
            #     cv2.imshow('gradient_costmap', gradient_costmap.get())
            # except Exception as e:
            #     print(f"Error displaying image: {e}")

            # gradient_costmap = apply_gradient_inflation_kernel(resized_costmap_cp)
            euclidean_costmap_cp = euclidean_costmap(gradient_costmap)

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

            # print(f"target = {self.target} tar = {targ}")

            euclidean_costmap_targ_cp = euclidean_costmap_from_target(euclidean_costmap_cp, target_cp=targ)
            try:
                cv2.imshow('euclidean_costmap_targ_cp', euclidean_costmap_targ_cp.get())
            except Exception as e:
                print(f"Error displaying image: {e}")
            # print(f"Max value in costmap before thresholding: {euclidean_costmap_targ_cp.max()}")
            # print(f"Min value in costmap before thresholding: {euclidean_costmap_targ_cp.min()}")

            # # Call the GPU function and capture the returned array.

            path_etched_costmap = reduce_min_cost_on_gpu(euclidean_costmap_targ_cp,1)
            try:
                cv2.imshow('path_etched_costmap', path_etched_costmap.get())
            except Exception as e:
                print(f"Error displaying image: {e}")
            

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
