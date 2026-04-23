import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ur_type = LaunchConfiguration('ur_type', default='ur7e')
    launch_rviz = LaunchConfiguration('launch_rviz', default='true')

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py',
            )
        ),
        launch_arguments={
            'pointcloud.enable': 'true',
            'rgb_camera.color_profile': '1920x1080x30',
        }.items(),
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur_moveit_config'),
                'launch',
                'ur_moveit.launch.py',
            )
        ),
        launch_arguments={'ur_type': ur_type, 'launch_rviz': launch_rviz}.items(),
    )

    return LaunchDescription(
        [
            realsense,
            Node(package='tidyer', executable='process_pointcloud', output='screen'),
            Node(package='tidyer', executable='tidyer_tf', output='screen'),
            moveit,
        ]
    )
