import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('description')

    model_path = os.path.join(pkg, 'urdf', 'puzzlebot_mcr2.urdf.xacro')
    rviz_config = os.path.join(pkg, 'rviz', 'week1.rviz')

    robot_description = ParameterValue(
        Command(['xacro ', model_path]),
        value_type=str
    )

    # Publishes robot TF tree (base_footprint -> all child links)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
    )

    # Publishes odom -> base_footprint TF (circular path) + spinning wheel joint states
    circular_motion = Node(
        package='description',
        executable='circular_motion.py',
    )

    # RViz with odom as fixed frame so the robot visibly moves around the origin
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
    )

    return LaunchDescription([
        robot_state_publisher,
        circular_motion,
        rviz,
    ])
