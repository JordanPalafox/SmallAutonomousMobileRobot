"""
robot.launch.py — ON-BOARD (Jetson Nano) half of the distributed real-robot
stack. Pairs with laptop.launch.py.

Design rule (split with laptop.launch.py):
  * The Jetson runs the whole SLAM sensing loop locally, so the LiDAR /scan
    and wheel /odom never cross WiFi before they are matched. That keeps the
    scan↔odom latency budget (< 10 ms, §2) intact.
  * The laptop runs everything else (nav, mission, dashboard, voice, RViz,
    map saver) so the 2 GB Jetson is not loaded — see laptop.launch.py.

Runs here, on the robot:
  controller/real       — velocity_bridge + real_odom + vel_smoother + RSP
                          (wheel odom for SLAM + base→lidar TF + cmd path)
  slam_node             — C++ front-end (GPU MCL) + back-end thread (graph
                          + loop closure + re-mapping). start_mode=navigation
                          (default): loads the saved map (map_yaml) and
                          LOCALISES against it without rebuilding it.
                          start_mode=mapping: builds + publishes /map live.
                          nav on the laptop plans over the published /map.
  map_saver             — `/map_saver/save_map` Trigger → writes the .pgm/.yaml
                          to THIS host's ~/ros2_maps/warehouse, the exact path
                          slam_node reloads in navigation mode (so re-map →
                          save → relaunch needs no scp). Lives here, NOT on the
                          laptop (one node owns the /map_saver service).
  lifting_node          — FPGA lifter control over SPI (Tang Nano 20K on
                          /dev/spidev0.0). Subscribes /lifter_level (UInt8 0-3).
                          Must run on the Jetson — it owns the SPI hardware.

The LiDAR driver is NOT started here — it already runs as its own node,
publishing /scan at 10 Hz. Just make sure that scan's frame_id is `lidar_link`
(the URDF laser frame published by RSP), or add a static transform from the
driver's frame to lidar_link, so slam_node's base→laser TF lookup resolves.

Pre-conditions on the Jetson (started separately, hardware-specific):
  * LiDAR node publishing /scan @ 10 Hz (frame_id = lidar_link).
  * micro-ROS agent bridging the MCR2 hackerboard
      ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyTHS1
    (provides /VelocityEncL, /VelocityEncR, consumes /cmd_vel)
  * Clocks synced with the laptop (chrony); same ROS_DOMAIN_ID on both.
  * Headless — NO RViz here (run it on the laptop).

Live mapping → waypoints → navigation flow (no map save required):
  1. Launch this on the Jetson + laptop.launch.py on the laptop.
  2. Drive (teleop) to build the map — it streams to RViz live.
  3. Define waypoints on that live map from the dashboard / RViz; nav
     hot-reloads them (no restart).
  4. Switch the dashboard to NAVIGATION and send goals — A* plans over the
     live /map. Optionally save the finished map with the map_saver service.

Usage (on the Jetson):
  ros2 launch bringup robot.launch.py
  ros2 launch bringup robot.launch.py scan_time_offset:=0.05
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_controller  = get_package_share_directory('controller')
    pkg_slam        = get_package_share_directory('slam')
    pkg_lifting     = get_package_share_directory('lifting')
    pkg_perception  = get_package_share_directory('perception')

    slam_params_path    = os.path.join(pkg_slam, 'config', 'slam_params.yaml')
    lifting_params_path = os.path.join(pkg_lifting, 'config', 'lifting_params.yaml')
    camera_params_path  = os.path.join(pkg_perception, 'config', 'camera_params.yaml')

    map_yaml_default = os.path.expanduser('~/ros2_maps/warehouse.yaml')

    # ── Args ──────────────────────────────────────────────────────────
    # start_mode drives whether slam_node LOADS the saved map and only
    # localises (navigation) or builds a fresh one (mapping).  Keep this in
    # sync with the laptop.launch.py start_mode you pass on the other host.
    start_mode_arg = DeclareLaunchArgument(
        'start_mode', default_value='navigation',
        description='mapping | navigation. navigation (default): slam_node '
                    'loads map_yaml and localises against it WITHOUT rebuilding '
                    'it (falls back to live mapping if the file is missing). '
                    'mapping: build a fresh map from an empty grid.')
    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml', default_value=map_yaml_default,
        description='Saved map yaml slam_node localises against in navigation '
                    'mode. Pass map_yaml:="" to always map fresh.')
    start_mode = LaunchConfiguration('start_mode')
    map_yaml   = LaunchConfiguration('map_yaml')

    scan_time_offset_arg = DeclareLaunchArgument(
        'scan_time_offset', default_value='0.10',
        description='Seconds subtracted from /scan stamps to compensate the '
                    'LiDAR capture-to-delivery latency before odom lookup (§2). '
                    'The real RPLidar A1 is ~100-150 ms latent — leaving this 0 '
                    'gives double walls on every turn. Tune 0.05-0.15 to your '
                    'measured driver latency.')
    scan_time_offset = LaunchConfiguration('scan_time_offset')

    # The URDF only publishes the `lidar_link` frame, but the stock RPLidar A1
    # driver stamps scans with frame_id `laser`.  slam_node looks up
    # base_link -> <scan.frame_id> via TF and processes NOTHING until it
    # resolves (silent localisation death).  Bridge laser->lidar_link with an
    # identity static TF so SLAM works regardless of which name the driver uses.
    # Set bridge_laser_frame:=false if you configured the driver to publish
    # frame_id=lidar_link directly (then this bridge is unnecessary).
    bridge_laser_frame_arg = DeclareLaunchArgument(
        'bridge_laser_frame', default_value='true',
        description='Publish an identity lidar_link->laser static TF so SLAM '
                    'resolves the scan frame even if the driver stamps "laser".')

    # Lifter (FPGA over SPI). HAL comes from lifting_params.yaml (hal: spi,
    # spi_bus: 0). Set use_lifter:=false to skip it (e.g. bench runs with no
    # Tang Nano connected, to avoid the spidev open failing).
    use_lifter_arg = DeclareLaunchArgument(
        'use_lifter', default_value='true',
        description='Start the FPGA lifter node (SPI to the Tang Nano 20K).')

    # Logo-based PICK approach stop (perception/logo_stop_debug). Runs HERE so
    # the camera->stop decision has NO WiFi latency — the brownout fix: stop by
    # vision BEFORE crashing into the load instead of drive_until_stall. It
    # publishes /approach_stop/should_stop + /approach_stop/debug_image. Headless
    # on the Jetson; view the annotated image on the laptop with rqt_image_view.
    # Set logo_stop:=false to skip it (frees CPU on the 2 GB Nano when not
    # picking).
    logo_stop_arg = DeclareLaunchArgument(
        'logo_stop', default_value='true',
        description='Start the Electric-80 logo stop detector for the PICK '
                    'final approach (vision stop). false = do not start it.')

    # QR docking (qr_quad_alignment) runs HERE on the Jetson now (was on the
    # laptop): the docking control loop reads the camera + encoders LOCALLY, so
    # the pixel/distance feedback has no WiFi latency — the lag is what made the
    # laptop-side dock overshoot and crash into the load. The laptop SM only
    # sends /alignment_start and reads /alignment_state (small, latency-tolerant).
    # Disable on the laptop with qr:=false there to avoid two nodes fighting on
    # /cmd_vel_in.
    qr_arg = DeclareLaunchArgument(
        'qr', default_value='true',
        description='Run qr_quad_alignment (PICK docking) on the Jetson.')
    qr_dry_run_arg = DeclareLaunchArgument(
        'qr_dry_run', default_value='false',
        description='Run qr_quad_alignment WITHOUT driving /cmd_vel_in (testing).')
    qr_dock_dist_arg = DeclareLaunchArgument(
        'qr_dock_dist', default_value='0.33',
        description='DOCK stop distance to the QR in metres (height-agnostic). '
                    '0.0 = legacy pixel-cy mode. Calibrated 2026-06-03 at the '
                    'ideal dock pose: dist_qr=330mm, cx=173px, cy=187px (40/40 '
                    'detections). Read the live "d=..mm" in the dashboard to retune.')

    laser_frame_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_frame_bridge',
        arguments=['0', '0', '0', '0', '0', '0', 'lidar_link', 'laser'],
        condition=IfCondition(LaunchConfiguration('bridge_laser_frame')),
        output='screen',
    )

    # ── 1. Controller / odom / TF (hardware layer, must be local) ──────
    # velocity_bridge + real_odom + vel_smoother + robot_state_publisher.
    # real_odom must run here so /odom is time-synced with the local /scan.
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_controller, 'launch', 'real.launch.py')))

    # ── 2. SLAM core: localisation (navigation) or live mapping ────────
    # Front-end (GPU MCL) + back-end thread (graph/loop-closure/re-mapping).
    # start_mode=navigation → load map_yaml and localise WITHOUT rebuilding it
    # (the back-end map-rebuild + map-write are disabled in that mode);
    # start_mode=mapping → fresh empty grid, build + publish /map live.
    # /odom is the local real_odom topic.  Short delay so RSP + real_odom are
    # up before the first scan is consumed.
    slam_node = TimerAction(period=1.5, actions=[Node(
        package='slam', executable='slam_node', name='slam_node',
        parameters=[slam_params_path, {
            'use_sim_time':     False,
            'scan_time_offset': scan_time_offset,
            'map_yaml':         map_yaml,
            'start_mode':       start_mode,
        }],
        remappings=[('/odom', '/puzzlebot_controller/odom')],
        output='screen', emulate_tty=True,
    )])

    # ── 3. Map saver (lives WHERE slam loads the map) ──────────────────
    # Runs on the Jetson so the `/map_saver/save_map` Trigger (called manually
    # or auto-fired by the dashboard on MAPPING→NAVIGATION) writes the .pgm/
    # .yaml to THIS host's ~/ros2_maps/warehouse — exactly the path slam_node
    # reloads on the next navigation launch.  That closes the loop: re-map →
    # save → relaunch localises on the new map, with no scp to the Jetson.
    # It subscribes to the LOCAL /map (no WiFi), so saves are reliable.
    # (Keep it OFF the laptop — two `map_saver` nodes would clash on the
    # service name.  Waypoints still save on the laptop via the dashboard.)
    map_saver_node = Node(
        package='slam', executable='map_saver', name='map_saver',
        parameters=[{
            'use_sim_time': False,
            'map_path':     os.path.splitext(map_yaml_default)[0],
        }],
        output='screen',
    )

    # ── 4. Lifter (FPGA over SPI) ──────────────────────────────────────
    # Owns /dev/spidev0.0 → Tang Nano 20K. Independent of SLAM, no delay.
    lifting_node = Node(
        package='lifting', executable='lifting_node', name='lifting_node',
        parameters=[lifting_params_path],
        condition=IfCondition(LaunchConfiguration('use_lifter')),
        output='screen',
    )

    # ── 5. QR docking (local camera → no WiFi lag) ─────────────────────
    # Delayed 2 s so the camera + encoders are up. Headless (dashboard shows
    # /qr_quad_alignment/debug_image). dock_target_dist makes the stop distance
    # height-agnostic (works after lowering the QR).
    qr_node = TimerAction(period=2.0, actions=[Node(
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
            # Calibrated at the ideal dock pose (the QR sits right-of-centre, so
            # the old target_cx_px=160 never met tol and DOCK never reached DONE).
            'target_cx_px':        173.0,
            'target_cy_px':        187.0,
        }],
        condition=IfCondition(LaunchConfiguration('qr')),
        output='screen', emulate_tty=True,
    )])

    # ── 6. Logo stop detector (vision PICK approach stop) ──────────────
    # Delayed 3 s so it does not steal CPU from slam_node's MCL init on the
    # Nano. Calibrated defaults: template-only (mode 2), stop at apparent
    # scale 0.90 (before the reference distance ≈ contact, for margin).
    # Tune via stop_scale/match_thr; lower process_hz if CPU-bound.
    logo_stop_node = TimerAction(period=3.0, actions=[Node(
        package='perception', executable='logo_stop_debug', name='logo_stop_debug',
        parameters=[{
            'show_window':   False,
            'image_topic':   '/video_source/raw',
            'qos':           'sensor_data',
            'process_hz':    12.0,
            'mode':          2,
            # Stop well BEFORE the reference distance for margin: scales are
            # searched in 0.05 steps; 1.00 = reference ≈ contact, so 0.90 fires
            # two steps early (more clearance so the slow detector/creep never
            # rams the load → no brownout).
            'stop_scale':    0.90,
            'match_thr':     0.45,
            'hold_frames':   4,
            # ROI = bottom half only. The load's logo sits in the lower frame at
            # stop distance; the upper frame holds the rack POSTER (which also
            # carries E80 logos). Including it let a far/background logo match at
            # high scale → false "DONE while far". Excluding it kills that.
            'roi_top_pct':   50,
            'publish_debug': True,
        }],
        condition=IfCondition(LaunchConfiguration('logo_stop')),
        output='screen', emulate_tty=True,
    )])

    return LaunchDescription([
        start_mode_arg,
        map_yaml_arg,
        scan_time_offset_arg,
        bridge_laser_frame_arg,
        use_lifter_arg,
        logo_stop_arg,
        qr_arg,
        qr_dry_run_arg,
        qr_dock_dist_arg,
        laser_frame_bridge,
        controller_launch,
        slam_node,
        map_saver_node,
        lifting_node,
        logo_stop_node,
        qr_node,
    ])
