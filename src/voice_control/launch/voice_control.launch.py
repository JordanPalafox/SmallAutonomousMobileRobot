"""Launch file for the voice_control node (LPC+VQ voice recognition)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    mock_mode = LaunchConfiguration('mock_mode').perform(context)
    codebook_dir = LaunchConfiguration('codebook_dir').perform(context)

    pkg_voice = get_package_share_directory('voice_control')
    params_file = os.path.join(pkg_voice, 'config', 'voice_params.yaml')

    # Resolve default codebook directory inside the installed package
    if not codebook_dir:
        # At install time codebooks/ is not copied to share — point to the
        # source-space path if running from a colcon workspace.
        codebook_dir = os.path.join(pkg_voice, 'codebooks')

    voice_node = Node(
        package='voice_control',
        executable='voice_node',
        name='voice_node',
        parameters=[
            params_file,
            {
                'mock_mode': mock_mode.lower() in ('true', '1', 'yes'),
                'codebook_dir': codebook_dir,
            },
        ],
        output='screen',
    )

    return [voice_node]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'mock_mode',
            default_value='false',
            description=(
                'Set to "true" to subscribe to /mock_voice instead of using '
                'the real microphone.  Useful for integration testing.'
            ),
        ),
        DeclareLaunchArgument(
            'codebook_dir',
            default_value='',
            description=(
                'Absolute path to the codebooks/ directory.  '
                'Leave empty to use the package default.'
            ),
        ),
        OpaqueFunction(function=launch_setup),
    ])
