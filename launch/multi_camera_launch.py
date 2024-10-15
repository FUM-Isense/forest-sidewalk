from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
        # Path to the RealSense launch file
    realsense_launch_file_dir = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py'
    )
    # robot_localization_launch_file_dir = os.path.join(get_package_share_directory('robot_localization'), 'launch')

    return LaunchDescription([
        DeclareLaunchArgument('offline', default_value='false'),
        
        # Static transform for the first camera
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher_camera1',
            arguments=['0', '0', '-0.095', '0', '0', '0', 'base_link', 'camera1_link']
        ),

        # Static transform for the second camera
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher_camera2',
            arguments=['0.01', '0', '0.1', '0', '0', '0', 'base_link', 'camera2_link']
        ),

        # First camera - i
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file_dir),
            launch_arguments={
                'camera_name': 'camera1',
                'serial_no': '"139522075394"',
                'enable_sync': 'true',
                'align_depth.enable': 'true',
                'enable_color': 'true',
                'enable_gyro': 'true',
                'enable_accel': 'true',
                'unite_imu_method': '2',
                'pointcloud.enable': 'true',
                'pointcloud.allow_no_texture_points': 'true',
                'pointcloud.stream_filter': '0',
                'rgb_camera.color_profile': '640,480,30',
                'depth_module.depth_profile': '640,480,30',
                'temporal_filter.enable': 'true',
                'hole_filling_filter.enable': 'true',
                'camera_namespace': '/camera1'
            }.items(),
        ),


        # Second camera - f
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file_dir),
            launch_arguments={
                'camera_name': 'camera2',
                'serial_no': '"246422071801"',
                'enable_sync': 'true',
                'align_depth.enable': 'true',
                'enable_color': 'true',
                'enable_gyro': 'true',
                'enable_accel': 'true',
                'unite_imu_method': '2',
                'pointcloud.enable': 'true',
                'pointcloud.allow_no_texture_points': 'true',
                'pointcloud.stream_filter': '0',
                'rgb_camera.color_profile': '640,480,30',
                'depth_module.depth_profile': '640,480,30',
                'temporal_filter.enable': 'true',
                'hole_filling_filter.enable': 'true',
                'camera_namespace': '/camera2'
            }.items(),
        ),
        # RGBD Odometry for bot cameras
        Node(
            package='rtabmap_sync',
            executable='rgbd_sync',
            namespace='camera1',
            name='rgbd_1',
            output='screen',
            remappings=[
                ('rgb/image', '/camera1/camera1/color/image_raw'),
                ('depth/image', '/camera1/camera1/aligned_depth_to_color/image_raw'),
                ('rgb/camera_info', '/camera1/camera1/color/camera_info'),
            ],
        ),
        # RGBD Odometry for bot cameras
        Node(
            package='rtabmap_sync',
            executable='rgbd_sync',
            namespace='camera2',
            name='rgbd_2',
            output='screen',
            remappings=[
                ('rgb/image', '/camera2/camera2/color/image_raw'),
                ('depth/image', '/camera2/camera2/aligned_depth_to_color/image_raw'),
                ('rgb/camera_info', '/camera2/camera2/color/camera_info'),
            ],
        ),
        # RGBD Odometry for bot cameras
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='rtabmap_odometry',
            output='screen',
            remappings=[
                ('rgbd_image0', '/camera1/rgbd_image'),
                ('rgbd_image1', '/camera2/rgbd_image'),
                ('imu', '/camera1/camera1/imu'),
            ],
            parameters=[
                {'frame_id': 'base_link'},
                {'subscribe_rgbd': True},
                {'rgbd_cameras': 2},
                {'approx_sync': True},
                {'wait_imu_to_init': True}
            ]
        ),
        #  # RGBD Odometry for the first camera
        # Node(
        #     package='rtabmap_odom',
        #     executable='rgbd_odometry',
        #     namespace='camera1',
        #     name='rtabmap_odometry_camera1',
        #     output='screen',
        #     remappings=[
        #         ('rgb/image', '/camera1/camera1/color/image_raw'),
        #         ('depth/image', '/camera1/camera1/aligned_depth_to_color/image_raw'),
        #         ('rgb/camera_info', '/camera1/camera1/color/camera_info'),
        #         ('imu', '/camera1/camera1/imu'),
        #     ],
        #     parameters=[
        #         {'frame_id': 'camera1_link'},
        #         {'wait_imu_to_init': True},
        #         {'odom_frame_id': 'odom1'}
        #     ]
        # ),


        # # RGBD Odometry for the second camera
        # Node(
        #     package='rtabmap_odom',
        #     executable='rgbd_odometry',
        #     namespace='camera2',
        #     name='rtabmap_odometry_camera2',
        #     output='screen',
        #     remappings=[
        #         ('rgb/image', '/camera2/camera2/color/image_raw'),
        #         ('depth/image', '/camera2/camera2/aligned_depth_to_color/image_raw'),
        #         ('rgb/camera_info', '/camera2/camera2/color/camera_info'),
        #     ],
        #     parameters=[
        #         {'frame_id': 'camera2_link'},
        #         {'wait_imu_to_init': False},
        #         {'odom_frame_id': 'odom2'}
        #     ]
        # ),
    ])
