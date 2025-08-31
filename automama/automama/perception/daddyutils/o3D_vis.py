import open3d as o3d
import numpy as np
import cupy as cp

class Open3DVisualizer:
    def __init__(self, window_name="Point Cloud", width=960, height=540):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name, width=width, height=height)
        
        self.pcd = o3d.geometry.PointCloud()
        
        # New flag to track if geometry has been added
        self.geometry_added = False

    def update(self, points):

        # Update point cloud
        self.pcd.points = o3d.utility.Vector3dVector(points)

        # Only add the geometry if it's the first time and there are points
        if not self.geometry_added and points.shape[0] > 0:
            self.vis.add_geometry(self.pcd)
            self.geometry_added = True
        else:
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()