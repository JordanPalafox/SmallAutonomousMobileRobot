"""Launch file for the lifting node (FPGA lifter control)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    hal = LaunchConfiguration('hal').perform(context)

    pkg_lifting = get_package_share_directory('lifting')
    params_file = os.path.join(pkg_lifting, 'config', 'lifting_params.yaml')

    lifting_node = Node(
        package='lifting',
        executable='lifting_node',
        name='lifting_node',
        parameters=[
            params_file,
            {'hal': hal},
        ],
        output='screen',
    )

    return [lifting_node]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'hal',
            default_value='spi',
            description='HAL backend: mock | jetson | spi (Tang Nano 20K).',
        ),
        OpaqueFunction(function=launch_setup),
    ])
