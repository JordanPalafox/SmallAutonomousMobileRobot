from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true"
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    joy_teleop_node = Node(
        package="joy_teleop",
        executable="joy_teleop",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
    )

    return LaunchDescription([
        use_sim_time_arg,
        joy_node,
        joy_teleop_node,
    ])
