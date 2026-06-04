from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true"
    )
    use_simple_controller_arg = DeclareLaunchArgument(
        "use_simple_controller", default_value="true"
    )
    wheel_radius_arg = DeclareLaunchArgument(
        "wheel_radius", default_value="0.05"
    )
    wheel_separation_arg = DeclareLaunchArgument(
        "wheel_separation", default_value="0.19"
    )

    use_simple_controller = LaunchConfiguration("use_simple_controller")
    wheel_radius = LaunchConfiguration("wheel_radius")
    wheel_separation = LaunchConfiguration("wheel_separation")

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    lifter_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "lifter_controller",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    # Standard diff_drive_controller (when not using simple controller)
    wheel_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "puzzlebot_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        condition=UnlessCondition(use_simple_controller),
    )

    # Simple velocity controller + custom controller node
    simple_controller_group = GroupAction(
        condition=IfCondition(use_simple_controller),
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "simple_velocity_controller",
                    "--controller-manager",
                    "/controller_manager",
                ],
            ),
            Node(
                package="controller",
                executable="simple_controller",
                parameters=[{
                    "wheel_radius": wheel_radius,
                    "wheel_separation": wheel_separation,
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                }],
            ),
        ],
    )

    # Velocity smoother: rate-limits acceleration before forwarding to hardware
    twist_relay_node = Node(
        package="controller",
        executable="twist_relay",
        parameters=[{
            "use_sim_time":       LaunchConfiguration("use_sim_time"),
            "max_linear_accel":   0.3,   # m/s²  — tune to avoid current spikes
            "max_angular_accel":  0.5,   # rad/s²
            "rate":              50.0,   # Hz
        }],
        condition=IfCondition(use_simple_controller),
    )

    return LaunchDescription([
        use_sim_time_arg,
        use_simple_controller_arg,
        wheel_radius_arg,
        wheel_separation_arg,
        joint_state_broadcaster_spawner,
        lifter_controller_spawner,
        wheel_controller_spawner,
        simple_controller_group,
        twist_relay_node,
    ])
