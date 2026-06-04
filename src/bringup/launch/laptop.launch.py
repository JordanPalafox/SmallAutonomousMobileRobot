"""
laptop.launch.py — OFF-BOARD (laptop) half of the distributed real-robot
stack. Pairs with robot.launch.py running on the Jetson.

Design rule (split with robot.launch.py):
  * The Jetson runs the whole SLAM sensing loop locally (LiDAR + odom +
    front-end + back-end) so the scan never crosses WiFi before matching.
  * THIS launch runs everything else on the laptop CPU so the 2 GB Jetson
    stays free: planning, mission logic, the dashboard, voice, the map
    saver, and RViz. None of these are latency-critical against the scan.

Runs here, on the laptop:
  nav_node          — A* + Bug + pure-pursuit (consumes /map, TF, /scan)
  mission_control   — YASMIN state machine
  dashboard         — Flask + Socket.IO telemetry/web UI
  voice_control     — LPC + VQ recogniser (laptop microphone)
  robot_state_publisher + joint_state_publisher — local copy of the model/TF
  map_odom_relay    — rebuilds map->odom from /slam_pose
  rviz2             — visualisation (§7: RViz on the laptop, not the Nano)

What it does NOT run (those live on the Jetson, via robot.launch.py):
  rplidar driver, velocity_bridge, real_odom, vel_smoother, slam_node,
  map_saver.  The map_saver moved to the Jetson so `/map_saver/save_map`
  writes the .pgm/.yaml where slam_node reloads it (no scp); the dashboard's
  save button + MAPPING→NAVIGATION auto-save call that same global service
  over WiFi.  Only ONE map_saver may own the service, so it is NOT here.

The map, /scan, /odom and the Jetson's *static* TF all arrive over DDS/WiFi.
We DO run robot_state_publisher + joint_state_publisher locally because the
Jetson's *dynamic* /tf does not cross reliably: the movable joints (wheels,
mast — they need /joint_states, which nothing on the robot publishes) and
slam_node's map->odom (FastDDS multi-writer /tf quirk) both go missing. The
local RSP+JSP give RViz the full model, and map_odom_relay rebuilds map->odom
from /slam_pose (which does cross). Keep clocks synced (chrony), same
ROS_DOMAIN_ID, and the same FASTRTPS_DEFAULT_PROFILES_FILE on both hosts.

Usage (on the laptop):
  ros2 launch bringup laptop.launch.py
  ros2 launch bringup laptop.launch.py rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_slam = get_package_share_directory('slam')
    pkg_nav  = get_package_share_directory('navigation')
    pkg_desc = get_package_share_directory('description')
    pkg_perc = get_package_share_directory('perception')
    pkg_mc   = get_package_share_directory('mission_control')

    nav_params_path     = os.path.join(pkg_nav,  'config', 'nav_params.yaml')
    camera_params_path  = os.path.join(pkg_perc, 'config', 'camera_params.yaml')
    sm_params_path      = os.path.join(pkg_mc,   'config', 'sm_params.yaml')
    rviz_cfg_path    = os.path.join(pkg_slam, 'config', 'slam.rviz')
    urdf_path        = os.path.join(pkg_desc, 'urdf', 'puzzlebot_with_lifter.urdf.xacro')

    robot_description = ParameterValue(
        Command(['xacro ', urdf_path, ' is_sim:=false is_ignition:=false']),
        value_type=str)

    map_yaml_default       = os.path.expanduser('~/ros2_maps/warehouse.yaml')
    waypoints_yaml_default = os.path.expanduser('~/ros2_maps/waypoints.yaml')

    # ── Args ──────────────────────────────────────────────────────────
    start_mode_arg = DeclareLaunchArgument(
        'start_mode', default_value='navigation',
        description='Initial system mode for nav + dashboard: mapping | '
                    'navigation. Use start_mode:=mapping to build a fresh map '
                    'first (goals gated until you switch to navigation in the '
                    'dashboard). SLAM on the robot maps live regardless.')
    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml', default_value=map_yaml_default,
        description='Saved map yaml for nav fallback; empty = none.')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz2 with the SLAM config.')
    qr_arg = DeclareLaunchArgument(
        'qr', default_value='false',
        description='Launch qr_quad_alignment HERE on the laptop. Default false: '
                    'it now runs on the Jetson (robot.launch.py) so the docking '
                    'control loop reads the camera locally with no WiFi lag. Set '
                    'true only to fall back to laptop-side docking.')
    qr_dry_run_arg = DeclareLaunchArgument(
        'qr_dry_run', default_value='false',
        description='Run qr_quad_alignment WITHOUT driving /cmd_vel_in '
                    '(search-only testing; PICK docking needs false).')
    qr_dock_dist_arg = DeclareLaunchArgument(
        'qr_dock_dist', default_value='0.33',
        description='DOCK stop distance to the QR in metres (height-agnostic). '
                    'Calibrated 2026-06-03 at the ideal dock pose: dist_qr=330mm. '
                    'Set 0.0 for legacy pixel-cy mode. Read the live "d=..mm" in '
                    'the dashboard QR feed to retune.')

    start_mode = LaunchConfiguration('start_mode')
    map_yaml   = LaunchConfiguration('map_yaml')

    # ── Navigation ────────────────────────────────────────────────────
    nav_node = Node(
        package='navigation', executable='nav_node', name='nav_node',
        parameters=[nav_params_path, {
            'use_sim_time':   False,
            'map_yaml':       map_yaml,
            'waypoints_yaml': waypoints_yaml_default,
            'start_mode':     start_mode,
        }],
        output='screen', emulate_tty=True,
    )

    # ── Mission control ───────────────────────────────────────────────
    # IMPORTANT: load sm_params.yaml here. Without it the node falls back to the
    # in-code defaults (pick_approach_speed 0.10 instead of the tuned 0.04 — the
    # creep then races past the logo vision-stop and rams the load), and none of
    # the PICK tuning (creep speed, reverse time, vision-stop, dock) applies.
    mission_node = Node(
        package='mission_control', executable='state_machine_node',
        name='state_machine_node',
        parameters=[sm_params_path, {'use_sim_time': False}],
        output='screen',
    )

    # ── Dashboard ─────────────────────────────────────────────────────
    dashboard_node = Node(
        package='dashboard', executable='dashboard_node', name='dashboard_node',
        parameters=[{'use_sim_time': False, 'start_mode': start_mode}],
        output='screen',
    )

    # ── QR docking / detection ────────────────────────────────────────
    # Aligns onto the pallet QR (PICK) and publishes the annotated camera feed
    # (/qr_quad_alignment/debug_image) that the dashboard shows. Subscribes to
    # the Jetson camera (/video_source/raw) over DDS. show_window off (no cv2
    # window on launch); the dashboard is the viewer. marker_length matches
    # camera_params.yaml (0.05 m) for correct docking pose scale.
    qr_node = Node(
        package='perception', executable='qr_quad_alignment', name='qr_quad_alignment',
        parameters=[{
            'use_sim_time':        False,
            'image_topic':         '/video_source/raw',
            'camera_params':       camera_params_path,
            'marker_length':       0.05,
            'dry_run':             ParameterValue(LaunchConfiguration('qr_dry_run'), value_type=bool),
            'show_window':         False,
            'publish_debug_image': True,
            'dock_target_dist':    ParameterValue(LaunchConfiguration('qr_dock_dist'), value_type=float),
            # Calibrated at the ideal dock pose (QR right-of-centre at ideal).
            'target_cx_px':        173.0,
            'target_cy_px':        187.0,
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('qr')),
    )

    # ── Voice control ─────────────────────────────────────────────────
    voice_node = Node(
        package='voice_control', executable='voice_node', name='voice_node',
        parameters=[{'use_sim_time': False}],
        output='screen',
    )

    # ── Robot model + TF (local) ──────────────────────────────────────
    # The Jetson's dynamic /tf does not cross WiFi reliably (movable joints +
    # slam map->odom), so we publish the full model locally.
    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': False}],
    )
    joint_state_publisher = Node(
        package='joint_state_publisher', executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': False}],
    )
    # Rebuilds map->odom from /slam_pose (slam's own map->odom /tf doesn't cross).
    map_odom_relay = Node(
        package='controller', executable='map_odom_relay', name='map_odom_relay',
        parameters=[{'use_sim_time': False}],
        output='screen',
    )

    # ── RViz2 ─────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_cfg_path],
        parameters=[{'use_sim_time': False}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        start_mode_arg, map_yaml_arg, rviz_arg, qr_arg, qr_dry_run_arg, qr_dock_dist_arg,
        robot_state_publisher,
        joint_state_publisher,
        map_odom_relay,
        nav_node,
        mission_node,
        dashboard_node,
        qr_node,
        voice_node,
        rviz_node,
    ])
