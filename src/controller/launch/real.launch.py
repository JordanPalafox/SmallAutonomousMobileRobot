"""
real.launch.py — controller stack for the real Puzzlebot MCR2.

Does NOT use ros2_control or controller_manager (those are Gazebo-only).

Starts:
  real_odom    — integrates /wl + /wr velocities → /puzzlebot_controller/odom
                  + odom→base_footprint TF
  vel_smoother — rate-limits acceleration so the powerbank does not see a
                  current spike that reboots the hackerboard. Subscribes to
                  /cmd_vel_in, publishes ramped Twist to /cmd_vel (which the
                  micro_ros agent consumes) and TwistStamped to
                  puzzlebot_controller/cmd_vel (for the sim controller path).

All upstream nodes (navigation, perception, dashboard, teleop) must publish
to /cmd_vel_in — never directly to /cmd_vel.

Run alongside:
  ros2 launch slam mapping_real.launch.py   (provides velocity_bridge, RSP, SLAM)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg_desc = get_package_share_directory('description')
    urdf_path = os.path.join(pkg_desc, 'urdf', 'puzzlebot_with_lifter.urdf.xacro')
    robot_description = ParameterValue(
        Command(['xacro ', urdf_path, ' is_sim:=false is_ignition:=false']),
        value_type=str,
    )

    wheel_radius_arg = DeclareLaunchArgument(
        'wheel_radius', default_value='0.05',
        description='Wheel radius in metres')
    wheel_separation_arg = DeclareLaunchArgument(
        'wheel_separation', default_value='0.19',
        description='Wheel centre-to-centre distance in metres')
    max_linear_accel_arg = DeclareLaunchArgument(
        'max_linear_accel', default_value='0.6',
        description='Max linear acceleration m/s² — prevents current spikes')
    max_angular_accel_arg = DeclareLaunchArgument(
        'max_angular_accel', default_value='2.0',
        description='Max angular acceleration rad/s²')

    real_odom = Node(
        package='controller',
        executable='real_odom',
        name='real_odom',
        parameters=[{
            'wheel_radius':     LaunchConfiguration('wheel_radius'),
            'wheel_separation': LaunchConfiguration('wheel_separation'),
            'use_sim_time':     False,
        }],
        output='screen',
    )

    # Velocity smoother: sits between all cmd_vel_in publishers and the
    # hackerboard micro_ros agent (/cmd_vel).  Limits acceleration so the
    # power bank doesn't see a current spike that reboots the board.
    vel_smoother = Node(
        package='controller',
        executable='twist_relay',
        name='vel_smoother',
        parameters=[{
            'max_linear_accel':  LaunchConfiguration('max_linear_accel'),
            'max_angular_accel': LaunchConfiguration('max_angular_accel'),
            'rate':             50.0,
            'use_sim_time':     False,
        }],
        output='screen',
    )

    # Bridge encoder topics to /wl, /wr (replaces the old C++ velocity_bridge
    # from the deleted localization package).
    velocity_bridge = Node(
        package='controller',
        executable='velocity_bridge',
        name='velocity_bridge',
        output='screen',
    )

    # robot_state_publisher — publishes base_link → laser, lifter, etc. TFs
    # from the URDF.  Previously came from the localization package.
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': False}],
        output='screen',
    )

    return LaunchDescription([
        wheel_radius_arg,
        wheel_separation_arg,
        max_linear_accel_arg,
        max_angular_accel_arg,
        velocity_bridge,
        real_odom,
        vel_smoother,
        robot_state_publisher,
    ])
