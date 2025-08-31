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
                    if (costmap[neighbor_index] > 250) {
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
                    if (costmap[neighbor_index] > 250) {
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
                    if (costmap[neighbor_index] > 250) {
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
                    if (costmap[neighbor_index] > 250) {
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
                    if (costmap[neighbor_index] > 250) {
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

def apply_gradient_inflation_kernel(costmap_cp, h_max_steps=20, v_max_steps=20, d_max_steps=15):
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
        if (costmap[index] < 10) {
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

    # Calculate the dynamic start position as the mid-center bottom pixel
    start_x = cols // 2
    start_y = rows - 1
    # euclidean_factor = 1.0

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


class NavStack:
    """
    A navigation stack utility class that handles dynamic motion planning
    and command generation.

    This class first preprocesses the GPU-based costmap to a CPU-compatible
    format and then performs A* pathfinding to generate steering and throttle
    commands.
    """
    def __init__(self):
        """
        Initializes the navigation stack with a start and goal position.
        
        Args:
            start_pos (tuple): The (y, x) start coordinates.
            goal_pos (tuple): The (y, x) goal coordinates.
        """
        # self.costmap = None
        self.start = None
        self.target = None
        # self.rows = None
        # self.cols = None

    def preprocess_costmap(self, costmap_cp):
        """
        Takes a CuPy costmap and preprocesses it for CPU-based planning.

        This method first ensures the costmap is a 2D array, then moves it
        from the GPU to the CPU and stores it as a NumPy array. This is a
        critical step as the A* algorithm is a sequential search best suited
        for CPU execution.

        Args:
            costmap_cp (cupy.ndarray): The input costmap as a CuPy array.
        """
        if costmap_cp is None:
            return None 
        # new_costmap_cp = cp.where(costmap_cp == 0, 0, 255).astype(cp.uint8)

        # avg_coords_cp = cp.mean(cp.argwhere(costmap_cp == 0)[:10], axis=0)
        # print(avg_coords_cp.astype(cp.int16))
        self.target= cp.argwhere(costmap_cp == 0)[0]
        # self.target= avg_coords_cp.astype(cp.int16)
        self.start = cp.array([costmap_cp.shape[0] - 25, costmap_cp.shape[1] // 2], dtype=cp.int32)
        # print(f"Start pos: {self.start}, Target pos: {self.target}")
        costmap_vis = CostmapVisualizer.add_markers(costmap_cp,self.start,self.target,15)
        rows_with_zeros_mask = cp.any(costmap_cp == 0, axis=1)

        # Use the boolean mask to index the original array. This creates a new
        # array containing only the rows where the mask is True.
        resized_costmap_cp = costmap_cp[rows_with_zeros_mask]

        # cv2.imshow('resized cost',resized_costmap_cp.get())

        gradient_costmap = apply_gradient_inflation_kernel(resized_costmap_cp)
        # print(f"Preprocessed costmap shape: {preprocessed_resized_costmap_cp.shape}")
        cv2.imshow('gradient_costmap',gradient_costmap.get())

        euclidean_costmap_cp = euclidean_costmap(gradient_costmap)
        cv2.imshow('euclidean_costmap_cp',euclidean_costmap_cp.get())

        # print(cp.array_equal(gradient_costmap, euclidean_costmap_cp))
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
    
