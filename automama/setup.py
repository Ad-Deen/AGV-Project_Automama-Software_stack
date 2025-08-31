from setuptools import find_packages, setup
import os
from glob import glob # <--- Add this import

package_name = 'automama'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['automama*', 'daddyutils*'], exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add this line if you have launch files in a 'launch' directory:
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='deen',
    maintainer_email='deen@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'talker = automama.test.talker:main',
            'road_segmentation = automama.perception.road_segment:main',
            'cam_stream = automama.perception.live_cam_stream:main',
            'manual_control = automama.control.manual_control:main',
            'occupancy_grid = automama.navigation.rt_costmap:main',
            'data_logger = automama.interface.data_logger:main',
            'depth = automama.perception.depth:main',
            'navstack = automama.perception.stereodaddy7:main',
            'semantic_seg = automama.perception.semantic_seg:main',
        ],
    },
)