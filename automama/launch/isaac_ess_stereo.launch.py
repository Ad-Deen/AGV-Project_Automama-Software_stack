# # SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# # Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# # SPDX-License-Identifier: Apache-2.0

# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import ComposableNodeContainer
# from launch_ros.descriptions import ComposableNode


# def generate_launch_description():
#     # Only keep arguments that are still dynamic (like engine_file_path and threshold)
#     engine_file_path = LaunchConfiguration('engine_file_path')
#     engine_file_path_arg = DeclareLaunchArgument(
#         'engine_file_path',
#         # default_value='/home/deen/workspaces/isaac_ros-dev/isaac_ros_assets/models/dnn_stereo_disparity/dnn_stereo_disparity_v4.1.0_onnx/ess.engine', # You MUST provide this value
#         default_value='${ISAAC_ROS_WS:?}/isaac_ros_assets/models/dnn_stereo_disparity/dnn_stereo_disparity_v4.1.0_onnx/ess.engine',
#         description='The absolute path to the ESS engine plan.')

#     threshold = LaunchConfiguration('threshold')
#     threshold_arg = DeclareLaunchArgument(
#         'threshold',
#         default_value='0.0',
#         description='Threshold value ranges between 0.0 and 1.0 '
#                     'for filtering disparity with confidence.')

#     launch_args = [
#         engine_file_path_arg,
#         threshold_arg,
#     ]

#     # Directly specify the topic names you are publishing
#     left_raw_image_topic = '/csi_cam_left'
#     right_raw_image_topic = '/csi_cam_right'
#     left_info_topic = '/left/camera_info'
#     right_info_topic = '/right/camera_info'

#     # --- REMOVED H264 DECODER NODES as your images are raw ---

#     left_rectify_node = ComposableNode(
#         name='left_rectify_node',
#         package='isaac_ros_image_proc',
#         plugin='nvidia::isaac_ros::image_proc::RectifyNode',
#         parameters=[{
#             # IMPORTANT: Set these to the desired output resolution for rectified images.
#             # This should ideally match the input resolution of your ESS engine (e.g., 960x576).
#             'output_width': 960,
#             'output_height': 576,
#             'type_negotiation_duration_s': 5,
#         }],
#         remappings=[
#             ('image_raw', left_raw_image_topic),
#             ('camera_info', left_info_topic),
#             ('image_rect', 'left/image_rect'),
#             ('camera_info_rect', 'left/camera_info_rect')
#         ]
#     )

#     right_rectify_node = ComposableNode(
#         name='right_rectify_node',
#         package='isaac_ros_image_proc',
#         plugin='nvidia::isaac_ros::image_proc::RectifyNode',
#         parameters=[{
#             'output_width': 960,
#             'output_height': 576,
#             'type_negotiation_duration_s': 5,
#         }],
#         remappings=[
#             ('image_raw', right_raw_image_topic),
#             ('camera_info', right_info_topic),
#             ('image_rect', 'right/image_rect'),
#             ('camera_info_rect', 'right/camera_info_rect')
#         ]
#     )

#     # --- DISPARITY NODE USING NVIDIA SGM ACCELERATION ---
#     disparity_node = ComposableNode(
#         package='isaac_ros_stereo_image_proc',
#         plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
#         name='disparity',
#         namespace='',
#         remappings=[
#             ('left/image_rect', 'left/image_rect'),
#             ('right/image_rect', 'right/image_rect'),
#             ('left/camera_info', 'left/camera_info_rect'),
#             ('right/camera_info', 'right/camera_info_rect')
#         ],
#         parameters=[{
#             # Tune parameters here if needed
#             'max_disparity': 128.0,  # Try 128 or 256 for Orin
#             'backends': 'ORIN',    # ORIN backend for best Jetson performance
#             'confidence_threshold': 0.3,
#             'window_size': 6,
#             'num_passes': 2,
#             'p1': 8,
#             'p2': 32,
#             'p2_alpha': 0.8
#         }]
#     )

#     # --- OPTIONAL DEPTH NODE TO CONVERT DISPARITY TO METRIC DEPTH ---
#     disparity_to_depth_node = ComposableNode(
#         name='DisparityToDepthNode',
#         package='isaac_ros_stereo_image_proc',
#         plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode'
#     )

#     # --- ESS NODE (COMMENTED OUT) ---
#     # disparity_node = ComposableNode(
#     #     name='disparity',
#     #     package='isaac_ros_ess',
#     #     plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
#     #     parameters=[{
#     #         'engine_file_path': engine_file_path,
#     #         'threshold': threshold,
#     #         # These MUST match the resolution your ESS engine was built for.
#     #         'input_layer_width': 960,  # <-- VERIFY THIS AGAINST YOUR ENGINE FILE
#     #         'input_layer_height': 576, # <-- VERIFY THIS AGAINST YOUR ENGINE FILE
#     #         'type_negotiation_duration_s': 5,
#     #     }],
#     #     remappings=[
#     #         ('left/image_rect', 'left/image_rect'),
#     #         ('right/image_rect', 'right/image_rect'),
#     #         ('left/camera_info', 'left/camera_info_rect'),
#     #         ('right/camera_info', 'right/camera_info_rect'),
#     #     ]
#     # )

#     container = ComposableNodeContainer(
#         name='depth_container',
#         namespace='depth',
#         package='rclcpp_components',
#         executable='component_container_mt',
#         composable_node_descriptions=[
#             left_rectify_node,
#             right_rectify_node,
#             disparity_node,
#             disparity_to_depth_node
#         ],
#         output='screen'
#     )

#     return LaunchDescription(launch_args + [container])


# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'left_raw_image_topic',
            default_value='/csi_cam_left',
            description='Topic name for left raw CSI camera image'),
        DeclareLaunchArgument(
            'right_raw_image_topic',
            default_value='/csi_cam_right',
            description='Topic name for right raw CSI camera image'),
        DeclareLaunchArgument(
            'left_camera_info_topic',
            default_value='/left/camera_info',
            description='Topic name for left camera info'),
        DeclareLaunchArgument(
            'right_camera_info_topic',
            default_value='/right/camera_info',
            description='Topic name for right camera info'),
    ]

    left_raw_image_topic = LaunchConfiguration('left_raw_image_topic')
    right_raw_image_topic = LaunchConfiguration('right_raw_image_topic')
    left_camera_info_topic = LaunchConfiguration('left_camera_info_topic')
    right_camera_info_topic = LaunchConfiguration('right_camera_info_topic')

    # Rectify nodes
    left_rectify_node = ComposableNode(
        name='left_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
            'type_negotiation_duration_s': 5,
        }],
        remappings=[
            ('image_raw', left_raw_image_topic),
            ('camera_info', left_camera_info_topic),
            ('image_rect', 'left/image_rect'),
            # ('image_rect', 'infra/image_rect_raw_mono'),
            ('camera_info_rect', 'left/camera_info_rect'),
        ],
    )

    right_rectify_node = ComposableNode(
        name='right_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
            'type_negotiation_duration_s': 5,
        }],
        remappings=[
            ('image_raw', right_raw_image_topic),
            ('camera_info', right_camera_info_topic),
            ('image_rect', 'right/image_rect'),
            ('camera_info_rect', 'right/camera_info_rect'),
        ],
    )

    # Image format converters (BGR → NV12)
    left_format_node = ComposableNode(
        name='left_format_converter',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'encoding_desired': 'nv12',
            'image_width': 960,      # match your rectified output
            'image_height': 576
        }],
        remappings=[
            ('image_raw', 'left/image_rect'),
            ('image', 'left/image_rect_nv12'),
        ],
    )

    right_format_node = ComposableNode(
        name='right_format_converter',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'encoding_desired': 'nv12',
            'image_width': 960,
            'image_height': 576
        }],
        remappings=[
            ('image_raw', 'right/image_rect'),
            ('image', 'right/image_rect_nv12'),
        ],
    )

    disparity_node = ComposableNode(
        name='disparity_node',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
        parameters=[{
            'backends': 'ORIN',
            'max_disparity': 128.0,
            'min_disparity': 1.0,
        }],
        remappings=[
            ('left/image_rect', 'left/image_rect_nv12'),
            ('right/image_rect', 'right/image_rect_nv12'),
            ('left/camera_info', 'left/camera_info_rect'),
            ('right/camera_info', 'right/camera_info_rect'),
        ],
    )

    disparity_to_depth_node = ComposableNode(
        name='DisparityToDepthNode',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
        remappings=[
            ('disparity', 'disparity'),
            ('camera_info', 'left/camera_info_rect'),
        ]
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            left_rectify_node,
            right_rectify_node,
            # left_format_node,
            # right_format_node,
            # disparity_node,
            # disparity_to_depth_node,
        ],
        output='screen',
    )

    return launch.LaunchDescription(launch_args + [container])
