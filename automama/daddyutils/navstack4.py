import cupy as cp
import numpy as np
from queue import PriorityQueue
import cv2
import cupyx.scipy.ndimage as ndi

# ==============================Gradient inflation kernel and function ===============================
# --- The CUDA C++ Kernel Code ---
# This is a string containing the kernel function written in CUDA.
# It is compiled and run on the GPU by CuPy.
kernel_code_gradient_inflation = """
extern "C" __global__
void inflate_with_gradient_kernel(unsigned char* costmap,
                               int rows,
                               int cols,
                               int h_max_steps,
                               int v_max_steps,
                               int d_max_steps) {
    // Calculate a unique index for this thread in the 2D grid.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread index is within the bounds of the costmap.
    if (x < cols && y < rows) {
        int index = y * cols + x;
        
        // Do not modify existing obstacles (value 255).
        if (costmap[index] != 255) {
            // Initialize step counts to a value greater than their respective max steps.
            int steps_to_obstacle_left = h_max_steps + 1;
            int steps_to_obstacle_right = h_max_steps + 1;
            int steps_to_obstacle_up = v_max_steps + 1;
            int steps_to_obstacle_up_left = d_max_steps + 1;
            int steps_to_obstacle_up_right = d_max_steps + 1;

            // Search to the left from the current pixel up to h_max_steps
            for (int dx = 1; dx <= h_max_steps; ++dx) {
                int neighbor_x = x - dx;
                if (neighbor_x >= 0) {
                    int neighbor_index = y * cols + neighbor_x;
                    if (costmap[neighbor_index] > 230) {
                        steps_to_obstacle_left = dx;
                        break; 
                    }
                } else {
                    break;
                }
            }

            // Search to the right from the current pixel up to h_max_steps
            for (int dx = 1; dx <= h_max_steps; ++dx) {
                int neighbor_x = x + dx;
                if (neighbor_x < cols) {
                    int neighbor_index = y * cols + neighbor_x;
                    if (costmap[neighbor_index] > 230) {
                        steps_to_obstacle_right = dx;
                        break;
                    }
                } else {
                    break;
                }
            }

            // Search upwards from the current pixel up to v_max_steps
            for (int dy = 1; dy <= v_max_steps; ++dy) {
                int neighbor_y = y - dy;
                if (neighbor_y >= 0) {
                    int neighbor_index = neighbor_y * cols + x;
                    if (costmap[neighbor_index] > 230) {
                        steps_to_obstacle_up = dy;
                        break;
                    }
                } else {
                    break;
                }
            }

            // Search diagonally up-left up to d_max_steps
            for (int d = 1; d <= d_max_steps; ++d) {
                int neighbor_x = x - d;
                int neighbor_y = y - d;
                if (neighbor_x >= 0 && neighbor_y >= 0) {
                    int neighbor_index = neighbor_y * cols + neighbor_x;
                    if (costmap[neighbor_index] > 230) {
                        steps_to_obstacle_up_left = d;
                        break;
                    }
                } else {
                    break;
                }
            }

            // Search diagonally up-right up to d_max_steps
            for (int d = 1; d <= d_max_steps; ++d) {
                int neighbor_x = x + d;
                int neighbor_y = y - d;
                if (neighbor_x < cols && neighbor_y >= 0) {
                    int neighbor_index = neighbor_y * cols + neighbor_x;
                    if (costmap[neighbor_index] > 230) {
                        steps_to_obstacle_up_right = d;
                        break;
                    }
                } else {
                    break;
                }
            }
            
            // --- New Cost Calculation Logic ---
            float h_cost = 0.0f;
            float v_cost = 0.0f;
            float d_cost = 0.0f;
            int valid_cost_count = 0;

            // Calculate horizontal cost
            int min_h_steps = min(steps_to_obstacle_left, steps_to_obstacle_right);
            if (min_h_steps <= h_max_steps) {
                h_cost = 230.0f * ((float)(h_max_steps - min_h_steps) / h_max_steps);
                valid_cost_count++;
            }

            // Calculate vertical cost
            if (steps_to_obstacle_up <= v_max_steps) {
                v_cost = 230.0f * ((float)(v_max_steps - steps_to_obstacle_up) / v_max_steps);
                valid_cost_count++;
            }

            // Calculate diagonal cost
            int min_d_steps = min(steps_to_obstacle_up_left, steps_to_obstacle_up_right);
            if (min_d_steps <= d_max_steps) {
                d_cost = 230.0f * ((float)(d_max_steps - min_d_steps) / d_max_steps);
                valid_cost_count++;
            }

            // If any obstacle was found, calculate the new cost
            if (valid_cost_count > 0) {
                float average_cost = (h_cost + v_cost + d_cost) / 1;
                int new_cost = costmap[index] + (int)average_cost;
                costmap[index] = (unsigned char)((new_cost < 250) ? new_cost : 250);
            }
        }
    }
}
"""

def apply_gradient_inflation_kernel(costmap_cp, h_max_steps=10, v_max_steps=10, d_max_steps=10):
    """
    Applies a custom CUDA kernel to create a gradient-based cost around obstacles.

    Args:
        costmap_cp (cupy.ndarray): The input costmap as a CuPy array.
        h_max_steps (int): The max distance for horizontal searches.
        v_max_steps (int): The max distance for upward searches.
        d_max_steps (int): The max distance for diagonal searches.

    Returns:
        cupy.ndarray: The modified costmap.
    """
    rows, cols = costmap_cp.shape
    if rows == 0:
        print("no rows")
        return costmap_cp
    else:
        # Define the kernel launch configuration.
        threads_per_block = (16, 16)
        blocks_per_grid_x = (cols + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (rows + threads_per_block[1] - 1) // threads_per_block[1]

        # Load the new kernel
        gradient_inflation_kernel = cp.RawKernel(kernel_code_gradient_inflation, 'inflate_with_gradient_kernel')

        # Call the kernel with a 2D grid configuration.
        gradient_inflation_kernel(
            (blocks_per_grid_x, blocks_per_grid_y),
            threads_per_block,
            (costmap_cp, rows, cols, h_max_steps, v_max_steps, d_max_steps)
        )

        return costmap_cp

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
            int new_cost = costmap[index] + (int)euclidean_cost * 1.0; //scaling factor = 1.0
            
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
            int new_cost = costmap[index] + (int)(distance) *1.0; 
            
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
                costmap[min_indices[i]] = (unsigned char)((float)current_cost * 0.5f);
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


# =============================================================================




class NavStack:
    def __init__(self):
        self.start = None
        self.target = None
        self.prev_target = None
        self.x_threshold = 10

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
                    print(f"Filtered target found: {self.target.tolist()}")
                else:
                    # Fallback: If no targets are within the threshold
                    if potential_targets.shape == (1, 2):
                        self.target = potential_targets[0]
                    else:
                        self.target = potential_targets[0]
                    print(f"No targets within threshold, falling back to best available: {self.target.tolist()}")

            self.prev_target = self.target.tolist()
            # --- End of new target smoothing logic ---

            if (self.start is not None and self.start.ndim == 1 and self.start.shape[0] == 2 and
                self.target is not None and self.target.ndim == 1 and self.target.shape[0] == 2):
                costmap_vis = CostmapVisualizer.add_markers(costmap_cp, self.start, self.target, 15)
            else:
                print("Warning: Start or target position is not a valid coordinate pair. Skipping marker visualization.")
                costmap_vis = costmap_cp

            rows_with_zeros_mask = cp.any(costmap_cp == 0, axis=1)
            resized_costmap_cp = costmap_cp[rows_with_zeros_mask]
            gradient_costmap = apply_gradient_inflation_kernel(resized_costmap_cp)
            euclidean_costmap_cp = euclidean_costmap(gradient_costmap)

            try:
                cv2.imshow('euclidean_costmap_cp', euclidean_costmap_cp.get())
            except Exception as e:
                print(f"Error displaying image: {e}")

            if euclidean_costmap_cp is None:
                print("Euclidean costmap is None, returning from preprocess_costmap.")
                return None

            if euclidean_costmap_cp.size == 0:
                print("Warning: No suitable targets found after costmap processing (cost < 50).")
                return None
            else:
                # This line is now safe because the self.target variable is guaranteed to be a 1D array
                targ = [0, self.target[1]]

            print(f"target = {self.target} tar = {targ}")

            euclidean_costmap_targ_cp = euclidean_costmap_from_target(euclidean_costmap_cp, target_cp=targ)
            try:
                cv2.imshow('euclidean_costmap_targ_cp', euclidean_costmap_targ_cp.get())
            except Exception as e:
                print(f"Error displaying image: {e}")
            print(f"Max value in costmap before thresholding: {euclidean_costmap_targ_cp.max()}")
            print(f"Min value in costmap before thresholding: {euclidean_costmap_targ_cp.min()}")

            # Call the GPU function and capture the returned array.
            path_etched_costmap = reduce_min_cost_on_gpu(euclidean_costmap_targ_cp)
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
