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
    logo_model_default  = os.path.join(pkg_perc, 'config', 'weights.pt')
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
        'qr', default_value='true',
        description='Launch qr_quad_alignment HERE on the laptop (default true). '
                    'Moved off the Jetson: the QR detection at 640x360 pegged a '
                    'whole core on the 2 GB Nano (~105%) and saturated it, so it '
                    'runs on the laptop now. Trade-off: the docking control loop '
                    'reads the camera + cmd over WiFi (some lag). Set qr:=false on '
                    'the Jetson side (robot.launch.py) so only ONE runs.')
    qr_dry_run_arg = DeclareLaunchArgument(
        'qr_dry_run', default_value='false',
        description='Run qr_quad_alignment WITHOUT driving /cmd_vel_in '
                    '(search-only testing; PICK docking needs false).')
    qr_dock_dist_arg = DeclareLaunchArgument(
        'qr_dock_dist', default_value='0.28',
        description='ROLLER DOCK stop distance to the QR (m, height-agnostic). '
                    'Calibrated 2026-06-04 (QR "Popsi", 640x360): dist_qr=280mm. '
                    '(Rack uses rack_dock_target_dist=0.36 via /robot_state.) '
                    'Set 0.0 for legacy pixel-cy mode; read live "d=..mm" to retune.')
    logos_arg = DeclareLaunchArgument(
        'logos', default_value='true',
        description='Launch logo_classifier (YOLO) here. Publishes /logo_order '
                    '(left→right truck logos) which RELEASE_LOAD uses to deliver '
                    'the pallet to the matching truck. Set false to disable.')
    logo_model_arg = DeclareLaunchArgument(
        'logo_model', default_value=logo_model_default,
        description='Path to the YOLO weights.pt for logo_classifier. Default = '
                    'perception/config/weights.pt; override with an absolute path '
                    'if the model lives elsewhere.')

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
            # Roller (default) dock target — calibrated 2026-06-04 (QR "Popsi", 640x360).
            'target_cx_px':        325.6,
            'target_cy_px':        245.9,
            # RACK dock profile (mission 2: PICK_FROM_RACK), swapped by /robot_state.
            # MUST match robot.launch.py — this node now runs HERE (laptop), not on
            # the Jetson, to keep the 2 GB Nano from saturating.
            'rack_target_cx_px':     326.8,
            'rack_target_cy_px':     240.4,
            'rack_dock_target_dist': 0.36,
            'rack_kp_v_dock_dist':   0.6,   # >0.5 para no atascarse en el deadband (freeze)
            'rack_dock_max_linear':  0.04,  # un poco más rápido que el roller (0.035)
            'rack_dock_tol_cx_px':   15.0,
            'rack_kp_w_dock_px':     0.0022,
            'rack_kd_w_dock_px':     0.0011,
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('qr')),
    )

    # ── Logo classifier (YOLO) ────────────────────────────────────────
    # Recognises the 3 trailer logos (amazon/pepsi/walmart) in the Jetson
    # camera feed and publishes them left→right on /logo_order. RELEASE_LOAD
    # subscribes to that to deliver the pallet to the truck whose logo matches
    # the pallet's QR. Headless (show_window off); /logo_debug is the viewer.
    logo_classifier_node = Node(
        package='perception', executable='logo_classifier', name='logo_classifier',
        parameters=[{
            'use_sim_time': False,
            'model_path':   LaunchConfiguration('logo_model'),
            'image_topic':  '/video_source/raw',
            'show_window':  False,
            'publish_debug': True,
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('logos')),
    )

    # ── Logo-stop + centring for the pick approach (roller AND rack) ──
    # Runs HERE on the laptop (the camera already crosses for qr_node) so the
    # 2 GB Nano stays light. Per-pick-type LOGO profiles: the DEFAULT params are
    # the ROLLER (template e80_logo_ref.png, used during PICK); a RACK profile
    # (rack_* params, template e80_logo_ref_rack.png — the foreshortened top-view
    # logo) is selected when /robot_state == PICK_FROM_RACK. Gated by
    # active_states to the two PICK states (only matches + publishes then; skips
    # the heavy matchTemplate otherwise). Publishes /approach_stop/should_stop
    # (logo-stop) + /approach_stop/center_error (lateral logo error → the SM
    # centres the creep so the lifter enters straight).
    logo_stop_arg = DeclareLaunchArgument(
        'logo_stop', default_value='true',
        description='Run logo_stop_debug here (roller + rack logo profiles, '
                    'gated to PICK / PICK_FROM_RACK) for the pick approach-stop + centring.')
    logo_stop_node = Node(
        package='perception', executable='logo_stop_debug', name='logo_stop_debug',
        parameters=[{
            'use_sim_time':  False,
            'image_topic':   '/video_source/raw',
            'qos':           'sensor_data',
            'show_window':   False,
            'process_hz':    12.0,
            'mode':          2,             # template multiescala
            'active_states': 'PICK,PICK_FROM_RACK',
            'rack_state':    'PICK_FROM_RACK',
            # ROLLER profile (default) — used during PICK.
            'template_path': os.path.join(pkg_perc, 'config', 'e80_logo_ref.png'),
            # 0.78 (was 0.85): más margen para que el match de escala del logo se
            # considere exitoso aunque el logo se vea algo más chico de lo ideal.
            'stop_scale':    0.78,
            'match_thr':     0.45,
            'roi_top_pct':   50,
            'hold_frames':   4,
            'publish_debug': True,
            # RACK profile — used during PICK_FROM_RACK (foreshortened top view).
            'rack_template_path': os.path.join(pkg_perc, 'config', 'e80_logo_ref_rack.png'),
            'rack_stop_scale':    0.95,
            'rack_match_thr':     0.40,
            'rack_roi_top_pct':   30,
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('logo_stop')),
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
        logos_arg, logo_model_arg, logo_stop_arg,
        robot_state_publisher,
        joint_state_publisher,
        map_odom_relay,
        nav_node,
        mission_node,
        dashboard_node,
        qr_node,
        logo_classifier_node,
        logo_stop_node,
        voice_node,
        rviz_node,
    ])
