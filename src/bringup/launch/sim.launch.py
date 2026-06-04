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
    DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_desc = get_package_share_directory('description')
    pkg_slam = get_package_share_directory('slam')
    pkg_nav  = get_package_share_directory('navigation')

    map_yaml_default       = os.path.expanduser('~/ros2_maps/warehouse.yaml')
    waypoints_yaml_default = os.path.expanduser('~/ros2_maps/waypoints.yaml')
    slam_params_path       = os.path.join(pkg_slam, 'config', 'slam_params.yaml')
    nav_params_path        = os.path.join(pkg_nav,  'config', 'nav_params.yaml')
    rviz_cfg_path          = os.path.join(pkg_slam, 'config', 'slam.rviz')

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
        }],
        remappings=[('/odom', '/puzzlebot_controller/odom')],
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
        world_name_arg,
        start_mode_arg,
        map_yaml_arg,
        rviz_arg,
        gazebo_launch,
        slam_node,
        map_saver_node,
        nav_node,
        mission_node,
        dashboard_node,
        rviz_node,
    ])
