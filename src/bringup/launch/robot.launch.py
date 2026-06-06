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
  * Camera driver publishing /video_source/raw @ ~10 Hz (640x360). aruco_localization,
    logo_stop_debug (and qr if enabled) depend on it — if it's dead they silently
    never detect (no error, just no /aruco_pose_estimate / no anchor).
  * micro-ROS agent bridging the MCR2 hackerboard
      ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyTHS1
    (provides /VelocityEncL, /VelocityEncR, consumes /cmd_vel)
  * FastDDS profile ~/.ros/fastrtps_profiles.xml (robot-subnet 192.168.137.x whitelist)
    present on BOTH Jetson and laptop, with FASTRTPS_DEFAULT_PROFILES_FILE pointing to
    it. Do NOT clear it here (only sim.launch.py clears it for off-network localhost).
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


def _flat_obstacles(path):
    """Lee static_layout_*.yaml y lo aplana a [cx,cy,sx,sy,yaw_rad]×N para el param
    aruco_static_obstacles del slam_node. [] si no existe."""
    import math
    import yaml
    try:
        d = yaml.safe_load(open(path))
        flat = []
        for o in d.get('obstacles', []):
            flat += [float(o['x']), float(o['y']), float(o['sx']), float(o['sy']),
                     math.radians(float(o.get('yaw_deg', 0.0)))]
        return flat
    except Exception:
        return []


def generate_launch_description():
    pkg_controller  = get_package_share_directory('controller')
    pkg_slam        = get_package_share_directory('slam')
    pkg_lifting     = get_package_share_directory('lifting')
    pkg_perception  = get_package_share_directory('perception')

    slam_params_path    = os.path.join(pkg_slam, 'config', 'slam_params.yaml')
    lifting_params_path = os.path.join(pkg_lifting, 'config', 'lifting_params.yaml')
    camera_params_path  = os.path.join(pkg_perception, 'config', 'camera_params.yaml')
    aruco_map_path      = os.path.join(pkg_perception, 'config', 'aruco_map.yaml')
    # Layout estático del gemelo digital en frame REAL (best-fit del .world sobre los
    # marcadores reales, ~0.12 m rmse) → slam_node lo estampa al anclar.
    static_layout_real = _flat_obstacles(
        os.path.join(pkg_perception, 'config', 'static_layout_real.yaml'))

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

    # Pose inicial del robot en el mapa anclado (frame `map`, esquina SW = 0,0).
    # En NAVEGACIÓN el slam carga el mapa anclado y siembra la creencia aquí; el
    # default (0,0,0) es la esquina SW. Si colocas el robot en otro punto conocido,
    # pásalo (p.ej. initial_x:=1.8 initial_y:=0.5 initial_theta:=1.57); si no, usa
    # "2D Pose Estimate" en RViz tras lanzar, o deja que el rescate ArUco lo recupere.
    initial_x_arg = DeclareLaunchArgument(
        'initial_x', default_value='0.0',
        description='X inicial del robot en el mapa anclado [m] (SW corner = 0,0).')
    initial_y_arg = DeclareLaunchArgument(
        'initial_y', default_value='0.0',
        description='Y inicial del robot en el mapa anclado [m] (SW corner = 0,0).')
    initial_theta_arg = DeclareLaunchArgument(
        'initial_theta', default_value='0.0',
        description='Heading inicial del robot en el mapa anclado [rad].')

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
        'qr', default_value='false',
        description='Run qr_quad_alignment (PICK docking) on the Jetson. Default '
                    'FALSE now: at 640x360 the QR detection pegged a whole core '
                    '(~105%) and saturated the 2 GB Nano, so docking runs on the '
                    'laptop (laptop.launch.py qr:=true). Set qr:=true here only to '
                    'go back to on-Jetson docking (no WiFi lag, but heavy CPU).')
    qr_dry_run_arg = DeclareLaunchArgument(
        'qr_dry_run', default_value='false',
        description='Run qr_quad_alignment WITHOUT driving /cmd_vel_in (testing).')
    qr_dock_dist_arg = DeclareLaunchArgument(
        'qr_dock_dist', default_value='0.28',
        description='DOCK stop distance to the QR in metres (height-agnostic). '
                    '0.0 = legacy pixel-cy mode. Calibrated 2026-06-04 at the '
                    'ideal dock pose (QR "Popsi", 640x360): dist_qr=281mm, '
                    'cx=325.6px, cy=245.9px (50/50 detections). Read the live "d=..mm" '
                    'in the dashboard to retune.')

    # ArUco re-localisation. Runs HERE on the Jetson (local camera + local
    # slam_node) so the /aruco_pose_estimate → slam_node correction has NO WiFi
    # latency. Detecta los marcadores del piso y publica la pose absoluta del
    # robot en `map`; slam_node la usa para reposicionarse cuando deriva en
    # zonas ambiguas y deja que el scan la encaje con las paredes.
    aruco_arg = DeclareLaunchArgument(
        'aruco', default_value='true',
        description='Run aruco_localization (floor ArUco re-localisation).')
    aruco_pitch_arg = DeclareLaunchArgument(
        'aruco_cam_pitch_deg', default_value='20.0',
        description='Inclinación hacia ABAJO de la cámara [grados]. = el tilt del URDF '
                    '(cam_holder 20°), IGUAL que en sim (donde el single-marker funciona '
                    'bien). El single-marker depende de este valor; si la cámara real '
                    'está a otro ángulo, ajústalo (aruco_cam_pitch_deg:=X).')

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
            # Pista REAL = 3.65 (X) x 4.85 (Y) — AL REVÉS de los defaults SIM
            # (4.85x3.65) en slam_params.yaml. El chequeo de bounds del anclaje
            # (slam_node onArucoMarkers) rechaza poses fuera de [0..x]x[0..y]; sin
            # esto, los markers reales de la pared lejana en Y (id 7/13/17 a
            # y~4.5-4.85) caen fuera y el anclaje FASE-1 nunca cierra.
            'aruco_track_x':    3.65,
            'aruco_track_y':    4.85,
            # Pose inicial en el mapa anclado (esquina SW = 0,0). Default (0,0,0) =
            # esquina SW; si el robot NO arranca ahí, pasa initial_x/initial_y/
            # initial_theta, o usa "2D Pose Estimate" en RViz (o deja que el rescate
            # ArUco lo recupere al ver un marcador).
            'initial_x':        ParameterValue(LaunchConfiguration('initial_x'), value_type=float),
            'initial_y':        ParameterValue(LaunchConfiguration('initial_y'), value_type=float),
            'initial_theta':    ParameterValue(LaunchConfiguration('initial_theta'), value_type=float),
            # PAREDES EXTERNAS = fuente de verdad: al anclar (y al cargar el mapa
            # anclado) quema el perímetro canónico 3.65×4.85 como pared nítida,
            # encima del scan ruidoso del LiDAR.
            'aruco_stamp_walls':          True,
            'aruco_wall_thickness_cells': 2,
            # Gemelo digital (racks/rollers/camiones) en frame real → estampado al anclar.
            'aruco_static_obstacles':     static_layout_real,
            # Una vez que el scan encaja con la geometría canónica, deja el mapa FULL
            # LIMPIO (solo el gemelo) y localiza puro contra él.
            'aruco_full_clean':           True,
            # AJUSTES REAL para que el anclaje SÍ dispare en la pista (vs sim):
            #  - obs_max_range 3.0: espeja el max_range del nodo aruco (markers a ≤3 m).
            #  - anchor_cert 0.30: el MCL real tiene menos certeza que el sim.
            #  - reanchor_interval 8 s: re-rasterizar menos seguido (alivia el Nano 2 GB).
            'aruco_obs_max_range':        3.0,
            'aruco_anchor_cert':          0.30,
            'aruco_reanchor_interval':    8.0,
            # SCAN-MATCH MÁS ESTRICTO (REAL): sigma_hit más chico = verosimilitud más
            # afilada. El gemelo digital (racks/rollers/camiones) ROMPE la simetría del
            # almacén casi-rectangular, así que afilar amplifica esa pequeña diferencia
            # entre la esquina correcta y la espejo → el MCL pesa más la correcta y
            # resiste derivar al rincón equivocado. (default sim 0.10 → 0.06.)
            'amcl_sigma_hit':             0.06,
            # RELOCALIZACIÓN POR ArUco: con el mapa YA fixeado y los markers bien medidos,
            # la pose ArUco se PRIORIZA. Cuando discrepa de la pose MCL >0.6 m durante 2
            # fixes ciertos seguidos (single O multi), RE-SIEMBRA el MCL ahí AUNQUE el scan
            # encaje (lo saca de la esquina espejo). 0.6 m separa deriva normal (<0.6) del
            # error de esquina (>>0.6); 2 fixes = relocaliza rápido sin saltar por 1 ruidoso.
            'aruco_disagree_dist':        0.6,
            'aruco_disagree_count':       2,
            # Reducción de ruido del RPLidar A1 (REAL). Seguras (no tocan el lock):
            #  - scan_quality_min: filtra retornos débiles/especulares del A1 (no-op
            #    si el driver no publica intensities). Sube/baja si filtra de más/menos.
            #  - display_l_occ/free: render más nítido (más hits para pintar pared) —
            #    solo display, no afecta el matcheo.
            #  - outlier_max_jump: descarta más puntos saltados.
            'scan_quality_min':  5.0,
            'display_l_occ':     3.0,
            'display_l_free':   -2.0,
            'outlier_max_jump':  0.20,
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
            # Calibrated 2026-06-04 at the ideal dock pose (QR "Popsi", 640x360,
            # 50/50 detections, near-centred: cx~325.6, bearing -2.6 deg).
            'target_cx_px':        325.6,
            'target_cy_px':        245.9,
            # RACK dock profile (mission 2: PICK_FROM_RACK). qr_quad_alignment
            # swaps to these when /robot_state == PICK_FROM_RACK: the rack QR is
            # lower in the frame and farther. Recalibrated 2026-06-05 at the ideal
            # rack dock pose (QR "Wolmar", 640x360, 40/40 detections, tight spread).
            'rack_target_cx_px':     326.8,
            'rack_target_cy_px':     240.4,   # bajado 10 px (250.4 -> 240.4)
            'rack_dock_target_dist': 0.36,
            # Dock CONTROL params set EQUAL to the roller (per user: "las del
            # roller están bien"). Only the TARGET (cx/cy/dist) stays rack-specific.
            # NOTE: kp_v=0.30 can re-introduce the freeze-short if the dock starts
            # far (>~0.41 m); tol_cx must be 15 with kp_w=0.0022 (deadband floor).
            'rack_kp_v_dock_dist':   0.30,
            'rack_dock_max_linear':  0.035,
            'rack_dock_tol_cx_px':   15.0,
            'rack_kp_w_dock_px':     0.0022,
            'rack_kd_w_dock_px':     0.0011,
        }],
        condition=IfCondition(LaunchConfiguration('qr')),
        output='screen', emulate_tty=True,
    )])

    # ── 6. Logo stop detector (vision PICK approach stop) ──────────────
    # Delayed 3 s so it does not steal CPU from slam_node's MCL init on the
    # Nano. Calibrated defaults: template-only (mode 2), stop at apparent
    # scale 0.85 (before the reference distance ≈ contact, for margin).
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
            # searched in 0.05 steps; 1.00 = reference ≈ contact. 0.78 gives the
            # scale match extra tolerance so the vision stop is considered
            # successful even when lighting/angle keep the apparent logo a bit
            # smaller than ideal (and more clearance so the slow detector/creep
            # never rams the load → no brownout).
            'stop_scale':    0.78,
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

    # ── 7. ArUco re-localisation (local camera → no WiFi lag) ──────────
    # Delayed 2 s so the camera driver is up. Detecta los ArUco del piso
    # (DICT_ARUCO_ORIGINAL 5x5, 9 cm), estima la pose del robot en `map` con el
    # aruco_map.yaml y la publica en /aruco_pose_estimate. slam_node (local) la
    # consume para reposicionarse. La cámara real publica en /video_source/raw.
    aruco_node = TimerAction(period=2.0, actions=[Node(
        package='perception', executable='aruco_localization', name='aruco_localization',
        parameters=[{
            'use_sim_time':   False,
            'image_topic':    '/video_source/raw',
            'camera_params':  camera_params_path,
            'aruco_map':      aruco_map_path,
            'marker_length':  0.09,
            'dictionary':     'original',
            'map_frame':      'map',
            # Convención in-plane de la imagen del marker = IGUAL que sim (90), donde el
            # single-marker funciona bien. (El método igual la auto-resuelve, pero se
            # deja explícita para que el overlay y todo cuadre como en sim.)
            'marker_inplane_deg': 90.0,
            # EXTRÍNSECOS DE CÁMARA = como el gemelo/URDF/sim (donde el single-marker
            # funciona): 10 cm adelante, 20 cm de alto, pitch 20° (cam_holder del URDF).
            # El single-marker DEPENDE de estos; si la cámara real difiere, ajusta el
            # arg aruco_cam_pitch_deg / cam_xyz.
            'cam_xyz':        [0.10, 0.0, 0.20],
            'cam_pitch_deg':  ParameterValue(LaunchConfiguration('aruco_cam_pitch_deg'), value_type=float),
            # ANCLAJE: el frame `map` del SLAM == frame canónico ArUco con la esquina
            # SW del almacén en (0,0). aruco_map.yaml ya está medido desde esa esquina,
            # así que el origen es [0,0,0] (NO el centro de la pista, como en el diseño
            # viejo). Así el stream /aruco_markers (canon), el overlay /aruco_ideal_map
            # y /aruco_pose_estimate quedan en el frame SW=(0,0) al que ancla el SLAM.
            'aruco_origin_in_map': [0.0, 0.0, 0.0],
            # Rango de detección para el ANCLAJE: 2.0 m era muy corto — en un almacén
            # de 3.65×4.85 m, desde el centro las paredes están a ~1.8-2.4 m, así que
            # los markers quedaban >2 m y se filtraban → no se acumulaban ≥3 distintos
            # → el anclaje nunca cerraba. 3.0 m permite juntar suficientes markers.
            'max_range':      3.0,
            'publish_debug_image': True,
        }],
        condition=IfCondition(LaunchConfiguration('aruco')),
        output='screen', emulate_tty=True,
    )])

    return LaunchDescription([
        start_mode_arg,
        map_yaml_arg,
        scan_time_offset_arg,
        initial_x_arg,
        initial_y_arg,
        initial_theta_arg,
        bridge_laser_frame_arg,
        use_lifter_arg,
        logo_stop_arg,
        qr_arg,
        qr_dry_run_arg,
        qr_dock_dist_arg,
        aruco_arg,
        aruco_pitch_arg,
        laser_frame_bridge,
        controller_launch,
        slam_node,
        map_saver_node,
        lifting_node,
        logo_stop_node,
        qr_node,
        aruco_node,
    ])
