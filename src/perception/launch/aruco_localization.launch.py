"""Launch de la capa de localizacion por ArUco.

Corre `aruco_localization` (deteccion + estimacion de pose del robot).
Como `publish_poses=true`, tambien publica /aruco_poses, asi que NO hace
falta correr aruco_pose_node en paralelo.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('perception')
    params_file = os.path.join(pkg, 'config', 'aruco_params.yaml')
    camera_params = os.path.join(pkg, 'config', 'camera_params.yaml')
    aruco_map = os.path.join(pkg, 'config', 'aruco_map.yaml')

    image_topic = LaunchConfiguration('image_topic').perform(context)

    node = Node(
        package='perception',
        executable='aruco_localization',
        name='aruco_localization',
        parameters=[
            params_file,
            {
                'image_topic': image_topic,
                'camera_params': camera_params,
                'aruco_map': aruco_map,
            },
        ],
        output='screen',
    )
    return [node]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/cam_img',
            description='Topic de la imagen de camara.',
        ),
        OpaqueFunction(function=launch_setup),
    ])
