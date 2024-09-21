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
                'rgb_camera.color_profile': '1280,720,30',
                'depth_module.depth_profile': '1280,720,30',
                'temporal_filter.enable': 'true',
                'hole_filling_filter.enable': 'true'
            }.items()
        ),

        # Launch RTAB-Map Odometry
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='rtabmap_odometry',
            output='screen',
            remappings=[
                ('rgb/image', '/camera/camera/color/image_raw'),
                ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
                ('rgb/camera_info', '/camera/camera/color/camera_info'),
                ('imu', '/camera/camera/imu'),
            ],
            parameters=[
                {'frame_id': 'camera_link'},
                {'wait_imu_to_init': True}
            ]
        ),
    ])
