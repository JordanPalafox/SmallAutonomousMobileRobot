"""
all_pc.launch.py — Single-machine real-robot stack for the Puzzlebot AMR
(everything on ONE computer). Use this only when you are NOT splitting the
load across the Jetson + laptop; for the normal distributed setup use
robot.launch.py (on the Jetson) + laptop.launch.py (on the laptop).

Brings up the whole stack:

  * Controller (velocity_bridge + real_odom + vel_smoother + RSP)
      vel_smoother ramps /cmd_vel_in → /cmd_vel so the powerbank does not
      brown out the hackerboard on velocity steps.  All upstream nodes
      must publish to /cmd_vel_in (already wired in nav/perception/dashboard).
  * SLAM (slam_node) — live mapping + localisation
  * map_saver — `/map_saver/save_map` Trigger service
  * Navigation (nav_node)
  * Mission control state machine
  * Dashboard (Flask + SocketIO)
  * Voice control (LPC + VQ)
  * RViz2

Pre-condition: LiDAR node on /scan + micro-ROS agent for the hackerboard.

Usage:
  ros2 launch bringup all_pc.launch.py
  ros2 launch bringup all_pc.launch.py start_mode:=mapping
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_controller    = get_package_share_directory('controller')
    pkg_slam          = get_package_share_directory('slam')
    pkg_nav           = get_package_share_directory('navigation')

    map_yaml_default       = os.path.expanduser('~/ros2_maps/warehouse.yaml')
    waypoints_yaml_default = os.path.expanduser('~/ros2_maps/waypoints.yaml')
    slam_params_path       = os.path.join(pkg_slam, 'config', 'slam_params.yaml')
    nav_params_path        = os.path.join(pkg_nav,  'config', 'nav_params.yaml')
    rviz_cfg_path          = os.path.join(pkg_slam, 'config', 'slam.rviz')

    # ── Args ──────────────────────────────────────────────────────────
    start_mode_arg = DeclareLaunchArgument(
        'start_mode',
        default_value='navigation',
        description='Initial system mode (mapping | navigation). Defaults to '
                    'navigation; pass start_mode:=mapping when building a fresh map.',
    )
    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml', default_value=map_yaml_default,
        description=(
            'Path to saved map yaml. Default = ~/ros2_maps/warehouse.yaml so '
            'in navigation mode SLAM preloads the existing map and localises '
            'against it (instead of wiping it), and nav_node has a fallback '
            'initial grid. Pass an empty string to force a fresh empty grid.'
        ),
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz2 with the SLAM config',
    )

    start_mode = LaunchConfiguration('start_mode')
    map_yaml   = LaunchConfiguration('map_yaml')

    # ── 1. Controller (real_odom + twist_relay) ───────────────────────
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_controller, 'launch', 'real.launch.py')
        ),
    )

    # ── 2. (REMOVED) ekf_localization include ─────────────────────────
    # The localization package was deleted.  velocity_bridge + RSP are
    # now started directly by the controller launch above.

    # ── 3. SLAM ───────────────────────────────────────────────────────
    # Short delay so real_odom + RSP are publishing TF before SLAM consumes
    # /scan; kept short on purpose because RViz uses `map` as fixed frame
    # and the `map→odom` TF is published by this node — a long delay forces
    # RViz into a "queue full / dropping LaserScan" backlog at startup.
    # /odom is remapped because the sole odometry source on real hw is now
    # real_odom (the duplicate-TF EKF was disabled), which publishes on
    # /puzzlebot_controller/odom.
    # start_mode=navigation → slam loads map_yaml and localises against it
    # WITHOUT rebuilding it (so a good saved map is not wiped); start_mode=
    # mapping → fresh empty grid, build the map live.
    slam_node = TimerAction(period=1.5, actions=[Node(
        package='slam',
        executable='slam_node',
        name='slam_node',
        parameters=[slam_params_path, {
            'use_sim_time': False,
            'map_yaml':     map_yaml,
            'start_mode':   start_mode,
        }],
        remappings=[('/odom', '/puzzlebot_controller/odom')],
        output='screen',
        emulate_tty=True,
    )])

    # ── 4. Map saver ──────────────────────────────────────────────────
    map_saver_node = TimerAction(period=2.0, actions=[Node(
        package='slam',
        executable='map_saver',
        name='map_saver',
        parameters=[{
            'use_sim_time': False,
            'map_path':     os.path.splitext(map_yaml_default)[0],
        }],
        output='screen',
    )])

    # ── 5. Navigation ─────────────────────────────────────────────────
    nav_node = TimerAction(period=2.0, actions=[Node(
        package='navigation',
        executable='nav_node',
        name='nav_node',
        parameters=[nav_params_path, {
            'use_sim_time':   False,
            'map_yaml':       map_yaml,
            'waypoints_yaml': waypoints_yaml_default,
            'start_mode':     start_mode,
        }],
        output='screen',
        emulate_tty=True,
    )])

    # ── 6. Mission control ────────────────────────────────────────────
    mission_node = TimerAction(period=2.0, actions=[Node(
        package='mission_control',
        executable='state_machine_node',
        name='state_machine_node',
        parameters=[{'use_sim_time': False}],
        output='screen',
    )])

    # ── 7. Dashboard ──────────────────────────────────────────────────
    dashboard_node = TimerAction(period=2.0, actions=[Node(
        package='dashboard',
        executable='dashboard_node',
        name='dashboard_node',
        parameters=[{
            'use_sim_time': False,
            'start_mode':   start_mode,
        }],
        output='screen',
    )])

    # ── 9. Voice control ──────────────────────────────────────────────
    voice_node = Node(
        package='voice_control',
        executable='voice_node',
        name='voice_node',
        parameters=[{'use_sim_time': False}],
        output='screen',
    )

    # ── 10. RViz2 (SLAM config) ───────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg_path],
        parameters=[{'use_sim_time': False}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        start_mode_arg,
        map_yaml_arg,
        rviz_arg,
        controller_launch,
        slam_node,
        map_saver_node,
        nav_node,
        mission_node,
        dashboard_node,
        # lifting_node,  # disabled until the FPGA-driven lifter is wired up
        voice_node,
        rviz_node,
    ])
