import open3d as o3d
import numpy as np

class Open3DVisualizer:
    def __init__(self, window_name="Point Cloud", width=960, height=540):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name, width=width, height=height)
        
        # Initialize an empty point cloud
        self.pcd = o3d.geometry.PointCloud()
        
        # State variable for the first frame
        self.first_frame = True
        
        # --- NEW: Initialize the sphere geometry ---
        self.sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        self.sphere.paint_uniform_color([1, 0, 0])  # Set sphere color to red
        
        # A variable to track the previous sphere position
        self.previous_sphere_pos = np.zeros(3)

    def update(self, points):
        """
        Updates the point cloud and a dynamic sphere in the visualizer.
        The sphere's position is dictated by the first coordinate in the point cloud.
        The visualizer now handles both empty and non-empty incoming point streams.
        """
        # --- Manually insert the desired 3D coordinate at the beginning ---
        # sphere_target_pos = np.asarray([[0.0, 0.0, 0.0]])
        
        # Determine the points to be displayed in the point cloud
        # if points.shape[0] > 0:
        #     updated_points = np.insert(points, 0, sphere_target_pos, axis=0)
        # else:
        #     # If the incoming point cloud is empty, the PCD will contain only the sphere's point
        #     updated_points = sphere_target_pos

        # self.pcd.points = o3d.utility.Vector3dVector(updated_points)

        # Update the sphere's position
        # translation_vector = sphere_target_pos.flatten() - self.previous_sphere_pos
        # self.sphere.translate(translation_vector)
        # self.previous_sphere_pos = sphere_target_pos.flatten()
        
        if self.first_frame:
            # First frame: add both geometries to the visualizer
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(self.sphere)
            self.first_frame = False
        else:
            # Subsequent frames: update geometries
            self.vis.update_geometry(self.pcd)
            self.vis.update_geometry(self.sphere)
        
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()