# my_camera_nodes.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Define the Node for csi_camera_stream
    # Make sure 'automama' is the correct package name and 'csi_camera_stream'
    # is the exact executable name defined in your setup.py's console_scripts.
    csi_camera_node = Node(
        package='automama',
        executable='cam_stream',
        name='csi_camera_stream_node', # A unique name for this node instance
        output='screen', # Show output in the terminal
        # You can add parameters or remappings here if needed for this specific node
        # For example, if your camera stream needs a specific device ID:
        # parameters=[{'device_id': '/dev/video0'}]
    )
    # RViz node
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', '/home/deen/ros2_ws/src/automama/stereo.rviz'],
        output='screen'
    )

    return LaunchDescription([
        csi_camera_node,
        rviz,
    ])