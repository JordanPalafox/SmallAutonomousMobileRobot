import os
import shutil
from os import pathsep
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
)
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# ─────────────────────────────────────────────────────────────────────────
#  ⚠  GAZEBO = VISUALIZATION / INTEGRATION SIM ONLY  (desktop, not the robot)
#
#  * Physics is rigid-body and SLIP-FREE.  The real Puzzlebot MCR2 slips ~12%
#    in rotation / ~4% in translation, so its wheel odometry drifts on turns.
#    SLAM's motion-noise + scan-match gains are tuned for that slip; validating
#    them against Gazebo's perfect odometry will look great here and DIVERGE on
#    the real robot.  Tune SLAM with `ros2 launch slam sim.launch.py`
#    (puzzlebot_sim) which models slip AND LiDAR latency.
#  * This launch runs full Gazebo + Ogre2 + Xvfb — it will NOT fit a 2 GB
#    Jetson Nano.  On the robot use bringup/robot.launch.py + laptop.launch.py.
# ─────────────────────────────────────────────────────────────────────────
def generate_launch_description():
    puzzlebot_description = get_package_share_directory("description")

    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(puzzlebot_description, "urdf", "puzzlebot_with_lifter.urdf.xacro"),
        description="Absolute path to robot urdf file"
    )

    world_name_arg = DeclareLaunchArgument(
        name="world_name",
        default_value="empty"
    )

    world_path = PathJoinSubstitution([
        puzzlebot_description,
        "worlds",
        PythonExpression(expression=["'", LaunchConfiguration("world_name"), "'", " + '.world'"])
    ])

    model_path = str(Path(puzzlebot_description).parent.resolve())
    model_path += pathsep + os.path.join(puzzlebot_description, 'models')

    ros_lib = "/opt/ros/humble/lib"
    ld_lib = os.environ.get("LD_LIBRARY_PATH", "")
    plugin_path = pathsep.join([ros_lib, ld_lib])

    gz_resource_path = SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", model_path)
    ign_resource_path = SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", model_path)
    # Mesa software GL via GLX — avoids AMD driver GL3Plus crash
    gl_software = SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1")

    ros_distro = os.environ["ROS_DISTRO"]
    is_ignition = "True" if ros_distro == "humble" else "False"

    robot_description = ParameterValue(
        Command([
            "xacro ",
            LaunchConfiguration("model"),
            " is_ignition:=", is_ignition
        ]),
        value_type=str
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True
        }]
    )

    ign_exec = shutil.which("ign") or "ign"

    # ── Render backend selection (GZ_RENDER env var) ──────────────────
    #   GZ_RENDER=nvidia   → Ogre2 renders the LiDAR/camera on the NVIDIA GPU
    #                        via EGL headless (no X needed).  Gazebo then uses a
    #                        DIFFERENT GL stack than RViz (which uses Mesa/AMD on
    #                        :0), so the two no longer deadlock on a shared Mesa
    #                        resource → RViz opens alongside Gazebo on this
    #                        dual-GPU laptop.  This is the default.
    #   GZ_RENDER=software → Mesa llvmpipe on Xvfb :99 (no GPU).  Robust fallback
    #                        for machines without a usable NVIDIA EGL device, but
    #                        it contends with RViz's GL on this machine (RViz won't
    #                        open).  Use with rviz:=false (headless backend).
    gz_render = os.environ.get("GZ_RENDER", "nvidia").lower()
    if gz_render == "nvidia":
        # NVIDIA EGL headless: surfaceless, no X display required for the render.
        render_env = (
            "__NV_PRIME_RENDER_OFFLOAD=1 "
            "__GLX_VENDOR_LIBRARY_NAME=nvidia "
            "__VK_LAYER_NV_optimus=NVIDIA_only "
            "__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json "
        )
        render_flags = "--headless-rendering "
    else:
        # Pure Mesa software (llvmpipe) on Xvfb :99 — never touches a GPU.
        render_env = (
            "DISPLAY=:99 "
            "LIBGL_ALWAYS_SOFTWARE=1 "
            "__GLX_VENDOR_LIBRARY_NAME=mesa "
            "GALLIUM_DRIVER=llvmpipe "
            "MESA_LOADER_DRIVER_OVERRIDE=llvmpipe "
            "__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json "
        )
        render_flags = ""

    # Server-only mode (-s): runs physics without the GUI.
    # The GUI can be connected separately with: ign gazebo -g
    # This avoids the SIGINT crash that occurs when the GUI's Qt/ogre2
    # rendering hits a fatal error with the AMD GPU driver.
    # Use a Xvfb virtual display (:99) so Ogre2's GL3Plus render engine uses Mesa
    # software GLX instead of the NVIDIA or AMD hardware driver.  The real display
    # (:0) has NVIDIA 580 drivers installed which ignore LIBGL_ALWAYS_SOFTWARE and
    # the AMD GPU segfaults on GL3Plus buffer upload.
    # Remove any STALE :99 lock/socket before starting Xvfb.  A previous run
    # that was killed (Ctrl-C, crash) leaves /tmp/.X99-lock behind; the next
    # Xvfb then fails to own :99 and Gazebo's Ogre2 render thread hangs forever
    # at "Sensors.cc: Waiting for init" → no /scan, controller never activates,
    # and the whole sim looks "stuck".  Cleaning the lock makes every launch
    # start from a fresh display.  (:99 is private to this sim, so this is safe.)
    xvfb = ExecuteProcess(
        cmd=["bash", "-c",
             "rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null; "
             "exec Xvfb :99 -screen 0 1280x1024x24"],
        output="screen",
        on_exit=None,
    )

    gazebo_server = ExecuteProcess(
        cmd=[
            "bash", "-c",
            (
                f"{render_env}"
                f"IGN_IP=127.0.0.1 "
                f"GZ_SIM_SYSTEM_PLUGIN_PATH={plugin_path} "
                f"IGN_GAZEBO_SYSTEM_PLUGIN_PATH={plugin_path} "
                f"GZ_SIM_RESOURCE_PATH={model_path} "
                f"IGN_GAZEBO_RESOURCE_PATH={model_path} "
                f"ruby {ign_exec} gazebo -s -r {render_flags}$0 -v 4 --force-version 6"
            ),
            world_path,
        ],
        output="screen",
        on_exit=None,
    )

    # Wait for the server to be ready before spawning the robot
    gz_spawn_entity = TimerAction(
        period=5.0,
        actions=[Node(
            package="ros_gz_sim",
            executable="create",
            output="screen",
            arguments=["-topic", "robot_description", "-name", "puzzlebot"],
        )]
    )

    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            "/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
            "/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
        ],
        remappings=[("/imu", "/imu/out")],
        additional_env={"IGN_IP": "127.0.0.1"},
    )

    # Fallback joint_state_publisher: publishes zero-position joint states so
    # robot_state_publisher can broadcast TF for all moveable joints.
    # gz_ros2_control v0.7.18 has a persistent service-unresponsive bug that prevents
    # joint_state_broadcaster from spawning; this node ensures /joint_states is always
    # available for visualization even when the controller_manager is broken.
    joint_state_publisher_node = TimerAction(
        period=8.0,
        actions=[Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            parameters=[{"use_sim_time": True}],
        )]
    )

    # Spawn joint_state_broadcaster first, then puzzlebot_controller 6s later.
    # The diff_drive_controller needs command interfaces to be registered by
    # gz_ros2_control before configure — staggering avoids ok=False configure failure.
    jsb_spawner = TimerAction(
        period=12.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster", "--unload-on-kill"],
                output="screen",
            ),
        ]
    )

    diff_drive_spawner = TimerAction(
        period=18.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["puzzlebot_controller", "--unload-on-kill"],
                output="screen",
            ),
        ]
    )

    # Relay /cmd_vel_in → /puzzlebot_controller/cmd_vel_unstamped.
    # Every upstream node (navigation, dashboard, perception, teleop) publishes
    # to /cmd_vel_in (same convention as the real robot, where twist_relay
    # consumes it).  In sim we forward it straight to the diff_drive controller,
    # whose own linear/angular acceleration limits provide the smoothing that
    # twist_relay provides on hardware.  Relaying /cmd_vel (the OLD input) left
    # /cmd_vel_in unconsumed, so the robot never moved under navigation.
    cmd_vel_relay = Node(
        package="topic_tools",
        executable="relay",
        arguments=["/cmd_vel_in", "/puzzlebot_controller/cmd_vel_unstamped"],
        output="screen",
    )

    return LaunchDescription([
        model_arg,
        world_name_arg,
        gz_resource_path,
        ign_resource_path,
        # NOTE: gl_software (LIBGL_ALWAYS_SOFTWARE=1) is intentionally NOT added
        # globally here — it would leak into RViz (started by the parent launch)
        # and break RViz's GPU GL on :0.  The Gazebo SERVER that actually needs
        # Mesa software GL already sets LIBGL_ALWAYS_SOFTWARE=1 inline in its own
        # bash command above, so removing the global setter changes nothing for
        # Gazebo while letting RViz use the real GPU.
        robot_state_publisher_node,
        xvfb,
        gazebo_server,
        gz_spawn_entity,
        gz_ros2_bridge,
        joint_state_publisher_node,
        jsb_spawner,
        diff_drive_spawner,
        cmd_vel_relay,
    ])
