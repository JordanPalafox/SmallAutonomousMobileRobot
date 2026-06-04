import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory("puzzlebot_description")

    model_path = os.path.join(pkg, "urdf", "lifter.urdf.xacro")
    rviz_config = os.path.join(pkg, "rviz", "lifter_rviz.rviz")

    robot_description = ParameterValue(
        Command(["xacro ", model_path]),
        value_type=str,
    )

    # Publishes /robot_description and TF from joint states
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
    )

    # GUI sliders for mast_inner_joint and carriage_joint prismatic joints
    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz,
    ])
