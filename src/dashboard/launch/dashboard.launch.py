"""Launch file for the AMR web dashboard node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    host_arg = DeclareLaunchArgument(
        "host",
        default_value="0.0.0.0",
        description="IP address for the Flask web server to bind to",
    )
    port_arg = DeclareLaunchArgument(
        "port",
        default_value="8080",
        description="TCP port for the Flask web server",
    )
    debug_arg = DeclareLaunchArgument(
        "debug",
        default_value="false",
        description="Enable Flask debug mode (do NOT use in production)",
    )

    dashboard_node = Node(
        package="dashboard",
        executable="dashboard_node",
        name="dashboard_node",
        output="screen",
        parameters=[
            {
                "host":  LaunchConfiguration("host"),
                "port":  LaunchConfiguration("port"),
                "debug": LaunchConfiguration("debug"),
            }
        ],
    )

    return LaunchDescription([
        host_arg,
        port_arg,
        debug_arg,
        dashboard_node,
    ])
