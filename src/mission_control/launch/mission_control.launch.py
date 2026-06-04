"""
mission_control.launch.py
--------------------------
Launch the YASMIN state machine node for the AMR warehouse robot.

Launch arguments
----------------
use_sim_time : bool  (default false)
    Set to true when running in Gazebo simulation.
params_file : str  (default: package config/sm_params.yaml)
    Path to the ROS2 parameter YAML file.

Usage
-----
    # Real hardware
    ros2 launch mission_control mission_control.launch.py

    # Simulation
    ros2 launch mission_control mission_control.launch.py use_sim_time:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_dir = get_package_share_directory("mission_control")
    default_params = os.path.join(pkg_dir, "config", "sm_params.yaml")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulated clock from /clock topic (Gazebo)",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Absolute path to the state machine parameter YAML file",
    )

    state_machine_node = Node(
        package="mission_control",
        executable="state_machine_node",
        name="state_machine_node",
        parameters=[
            LaunchConfiguration("params_file"),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        use_sim_time_arg,
        params_file_arg,
        state_machine_node,
    ])
