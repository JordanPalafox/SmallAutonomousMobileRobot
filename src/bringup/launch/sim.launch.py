"""
sim.launch.py — Single-command simulation stack for the Puzzlebot AMR.

Brings up everything needed to map, save waypoints, navigate, and view the
dashboard in Gazebo:

  * Gazebo (server-only) + URDF + diff-drive controller + cmd_vel relay
    (via description/launch/gazebo.launch.py — uses Xvfb + Mesa software GL
     to dodge the AMD/Ogre2 GL crash on this machine).
  * SLAM (slam_node) with runtime MAPPING ↔ NAVIGATION mode switching.
  * map_saver_node — exposes the `/map_saver/save_map` service so the
    dashboard can persist the current map to ~/ros2_maps/warehouse.yaml.
  * Navigation node (nav_node) — A* + Bug1 + path-follower + local costmap.
  * Mission control state machine.
  * Dashboard (Flask + SocketIO) — UI to toggle mode, capture waypoints,
    send goals, save map.
  * RViz2 with the slam.rviz config.

Usage:
  ros2 launch bringup sim.launch.py
  ros2 launch bringup sim.launch.py world_name:=warehouse
  ros2 launch bringup sim.launch.py start_mode:=mapping
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable,
    TimerAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _flat_obstacles(path):
    """Lee static_layout_*.yaml y lo aplana a [cx,cy,sx,sy,yaw_rad]×N (yaw en rad)
    para el param aruco_static_obstacles del slam_node. [] si no existe."""
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
    pkg_desc = get_package_share_directory('description')
    pkg_slam = get_package_share_directory('slam')
    pkg_nav  = get_package_share_directory('navigation')
    pkg_perception = get_package_share_directory('perception')

    map_yaml_default       = os.path.expanduser('~/ros2_maps/warehouse.yaml')
    waypoints_yaml_default = os.path.expanduser('~/ros2_maps/waypoints.yaml')
    slam_params_path       = os.path.join(pkg_slam, 'config', 'slam_params.yaml')
    nav_params_path        = os.path.join(pkg_nav,  'config', 'nav_params.yaml')
    rviz_cfg_path          = os.path.join(pkg_slam, 'config', 'slam.rviz')
    # Layout estático del gemelo digital (racks/rollers/camiones) → aplanado
    # [cx,cy,sx,sy,yaw_rad]×N para que slam_node lo estampe al anclar. SIM = frame del
    # .world (exacto). Generado por scripts/extract_static_layout.py.
    static_layout_sim = _flat_obstacles(
        os.path.join(pkg_perception, 'config', 'static_layout_sim.yaml'))

    # ── Args ──────────────────────────────────────────────────────────
    world_name_arg = DeclareLaunchArgument(
        'world_name', default_value='almacen_racks',
        description='Gazebo world to load (almacen_racks, empty, …)',
    )
    start_mode_arg = DeclareLaunchArgument(
        'start_mode',
        default_value='navigation' if os.path.exists(map_yaml_default) else 'mapping',
        description='Initial system mode (mapping | navigation). '
                    'Default: navigation if a saved map exists, else mapping.',
    )
    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml', default_value=map_yaml_default,
        description='Path to saved map yaml (empty string disables preload)',
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz2 alongside the stack',
    )

    world_name = LaunchConfiguration('world_name')
    start_mode = LaunchConfiguration('start_mode')
    map_yaml   = LaunchConfiguration('map_yaml')

    # ── 1. Gazebo + URDF + diff-drive controller ──────────────────────
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_desc, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world_name': world_name}.items(),
    )

    # ── 2. SLAM (mapping + localization in one node) ──────────────────
    # Gazebo + controller take ~12-18 s to be ready (controller spawners run
    # at t=12, t=18 inside gazebo.launch.py).  Delay SLAM so /scan and
    # /puzzlebot_controller/odom are flowing before SLAM starts integrating.
    slam_node = TimerAction(period=20.0, actions=[Node(
        package='slam',
        executable='slam_node',
        name='slam_node',
        # CPU AMCL in the Gazebo sim: RViz already uses the GPU for its 3D
        # view, and CUDA context init + per-scan scoring then contend with it
        # (slow first map→odom → the RobotModel takes ages to appear).  The
        # CPU path is ~10-17 ms/scan — comfortably inside the 10 Hz budget —
        # so the sim stays responsive.  The REAL robot (robot.launch.py, no
        # Gazebo/RViz competing for the GPU) keeps amcl_use_cuda:=true.
        parameters=[slam_params_path, {
            'use_sim_time':   True,
            'amcl_use_cuda':  False,
            # El lidar SIM no tiene latencia (timestamp exacto de gz), así que NO
            # hay que restarle nada al stamp. El 0.04 (para el lidar real) hace
            # que en los GIROS el scan se integre contra un odom viejo -> deriva.
            'scan_time_offset': 0.0,
            # CRÍTICO: pasarle el modo + el mapa al slam_node. Sin esto usa sus
            # defaults (start_mode=mapping, map_yaml="") y SIEMPRE mapea de cero,
            # ignorando el mapa guardado (en navegación lo "cargaba" el nav_node
            # para su planner, pero el slam_node publicaba uno fresco encima).
            # Con esto, en navigation/localization carga el .pgm y NO lo modifica
            # (localization-only → anchored_=true, no re-mapea ni re-ancla).
            'start_mode':     start_mode,
            'map_yaml':       map_yaml,
            # Gemelo digital como fuente de verdad: al anclar, estampa el perímetro
            # canónico (4.85×3.65) + los racks/rollers/camiones del .world (exacto en
            # sim) y protege esas paredes del ruido del scan.
            'aruco_stamp_walls':       True,
            'aruco_static_obstacles':  static_layout_sim,
            'aruco_full_clean':        True,   # tras encajar, mapa = solo gemelo (full limpio)
            # Pose inicial = spawn del robot en gz (gazebo.launch.py -x 0.8255 -y
            # 2.8927, yaw 0). El frame gz == frame canónico ArUco (esquina SW en
            # 0,0), así que en NAVEGACIÓN el robot arranca ya localizado sobre el
            # mapa anclado (sin esto la creencia parte de (0,0) ~3 m fuera y el
            # filtro podría no recuperarse). En MAPEO solo fija dónde nace el frame.
            'initial_x':      0.8255,
            'initial_y':      2.8927,
            'initial_theta':  0.0,
        }],
        remappings=[('/odom', '/puzzlebot_controller/odom')],
        output='screen',
        emulate_tty=True,
    )])

    # ── 2b. ArUco localization (cámara sim del mástil) ────────────────
    # Detecta los ArUco del mundo en /mast_camera/image_raw, estima la pose del
    # robot en `map` y publica /aruco_pose_estimate (lo consume slam_node) +
    # /aruco_localization/debug_markers (LÍNEA ArUco→robot para verificar en
    # RViz si localiza bien) + /aruco_localization/debug_image (detecciones).
    # Usa los intrínsecos del sensor sim (camera_params_sim.yaml).
    # PROVISIONAL: cam_xyz / cam_pitch_deg / aruco_origin_in_map se calibran
    # observando la línea de debug vs la pose real del robot en Gazebo.
    aruco_node = TimerAction(period=21.0, actions=[Node(
        package='perception',
        executable='aruco_localization',
        name='aruco_localization',
        parameters=[{
            'use_sim_time':        True,
            'image_topic':         '/mast_camera/image_raw',
            'qos':                 'sensor_data',
            'camera_params':       os.path.join(pkg_perception, 'config', 'camera_params_sim.yaml'),
            # Mapa en el frame del MUNDO Gazebo (generado desde aruco_markers.yaml,
            # validado Δpos=0 vs el .world). map==mundo => aruco_origin_in_map=[0,0,0].
            'aruco_map':           os.path.join(pkg_perception, 'config', 'aruco_map_sim.yaml'),
            'marker_length':       0.09,
            # Giro in-plane de la convención de imagen del marker en gz. Probar
            # 90/-90/180/0 viendo la esfera roja vs el robot (mi estimación: 90).
            'marker_inplane_deg':  90.0,
            'dictionary':          'original',
            'map_frame':           'map',
            'cam_xyz':             [0.10, 0.0, 0.20],
            'cam_pitch_deg':       20.0,   # = tilt de la cámara en el URDF (cam_holder 20°)
            # Frame canónico = frame MUNDO de los markers (aruco_map_sim.yaml ya
            # está en mundo). Con [0,0,0] el ArUco publica la pose del robot en el
            # frame mundo; slam_node ANCLA el mapa a ese frame cuando ya es
            # confiable, así el mapa queda fijo sin importar dónde arrancó.
            'aruco_origin_in_map': [0.0, 0.0, 0.0],
            'max_range':           4.0,
            'publish_debug_image': True,
        }],
        output='screen',
        emulate_tty=True,
    )])

    # ── 3. Map saver — provides /map_saver/save_map Trigger service ───
    map_saver_node = TimerAction(period=21.0, actions=[Node(
        package='slam',
        executable='map_saver',
        name='map_saver',
        parameters=[{
            'use_sim_time': True,
            'map_path':     os.path.splitext(map_yaml_default)[0],  # without extension
        }],
        output='screen',
    )])

    # ── 4. Navigation (A* + Bug1 + local costmap) ─────────────────────
    nav_node = TimerAction(period=22.0, actions=[Node(
        package='navigation',
        executable='nav_node',
        name='nav_node',
        parameters=[nav_params_path, {
            'use_sim_time':   True,
            'map_yaml':       map_yaml,
            'waypoints_yaml': waypoints_yaml_default,
            'start_mode':     start_mode,
        }],
        output='screen',
        emulate_tty=True,
    )])

    # ── 5. Mission control state machine ──────────────────────────────
    mission_node = TimerAction(period=22.0, actions=[Node(
        package='mission_control',
        executable='state_machine_node',
        name='state_machine_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )])

    # ── 6. Dashboard (Flask + SocketIO) ───────────────────────────────
    dashboard_node = TimerAction(period=22.0, actions=[Node(
        package='dashboard',
        executable='dashboard_node',
        name='dashboard_node',
        parameters=[{
            'use_sim_time':   True,
            'start_mode':     start_mode,
        }],
        output='screen',
    )])

    # ── 7. RViz2 ──────────────────────────────────────────────────────
    # Start RViz EARLY (t=2), BEFORE Gazebo's Ogre2 render thread initialises
    # (~t=12-15, after the robot spawns).  Diagnosis (in the user's session)
    # showed RViz hangs during its OWN GL/Ogre init — no /dev/dri fds open,
    # main thread parked in nanosleep — but ONLY while Gazebo's render is
    # already active.  It is a render-init contention, NOT a frame/timing
    # issue.  Letting RViz grab + finish its GL context first (it needs ~3-5 s
    # and the GPU is idle that early) should make it immune; the scene then
    # fills in as TF/map/scan arrive (RViz retries TF forever, so starting
    # before the data exists is fine).
    # Do NOT set MESA_GL_VERSION_OVERRIDE / LIBGL_ALWAYS_SOFTWARE here: RViz is
    # a normal GUI app on the logged-in session and must use the desktop GPU
    # natively (verified ~190 MB RSS, "OpenGl version: 4.6").
    rviz_node = TimerAction(period=2.0, actions=[Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg_path],
        parameters=[{'use_sim_time': True}],
        # RViz must use the real GPU on :0 (like the desktop).  This Mesa build
        # treats LIBGL_ALWAYS_SOFTWARE as "set = force software" regardless of
        # value (even "0" forces it), so the var must be UNSET, not "0".  We do
        # NOT set it here, and gazebo.launch.py no longer leaks it globally
        # (only its own server subprocess sets it inline), so RViz inherits no
        # LIBGL var and gets a native OpenGL 4.6 context.
        output='screen',
        condition=__import__('launch.conditions', fromlist=['IfCondition']).IfCondition(
            LaunchConfiguration('rviz')
        ),
    )])

    return LaunchDescription([
        # La simulación es 100% local: NO uses el perfil FastDDS del robot real
        # (~/.ros/fastrtps_profiles.xml), que tiene whitelist a la IP de la red
        # del robot (192.168.137.x). En otra red ese perfil filtra TODAS las
        # interfaces -> el descubrimiento DDS muere -> el robot no spawnea y no
        # hay /scan ni cámara. Vaciarlo + localhost-only deja DDS por defecto.
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', ''),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '1'),
        world_name_arg,
        start_mode_arg,
        map_yaml_arg,
        rviz_arg,
        gazebo_launch,
        slam_node,
        aruco_node,
        map_saver_node,
        nav_node,
        mission_node,
        dashboard_node,
        rviz_node,
    ])
