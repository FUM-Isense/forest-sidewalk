from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the RealSense launch file
    realsense_launch_file_dir = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py'
    )

    return LaunchDescription([
        # Include the RealSense camera launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file_dir),
            launch_arguments={
                'enable_sync': 'true',
                'align_depth.enable': 'true',
                'enable_color': 'true',
                'enable_gyro': 'true',
                'enable_accel': 'true',
                'unite_imu_method': '2',
                'rgb_camera.color_profile': '640,480,30',
                'depth_module.depth_profile': '640,480,30'
            }.items()
        ),
    ])
