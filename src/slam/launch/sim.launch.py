"""
sim.launch.py — Puzzlebot SLAM in the realistic 2D simulator.

  puzzlebot_sim  — raycast LiDAR (noise + dropouts, 7 Hz) + slipped odom
  slam_node      — pure C++ SLAM (AMCL + scan-match + scan-to-map)
  rviz2          — visualise map, particles, scan, ground truth

Drive the robot with:
  ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args -r cmd_vel:=/cmd_vel_in
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
    pkg = get_package_share_directory('slam')
    pkg_desc = get_package_share_directory('description')
    params = os.path.join(pkg, 'config', 'slam_params.yaml')
    rviz_cfg = os.path.join(pkg, 'config', 'slam.rviz')
    urdf = os.path.join(pkg_desc, 'urdf', 'puzzlebot_with_lifter.urdf.xacro')

    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')

    # Full robot model so RViz shows the 3D mesh.  is_sim:=false → URDF lidar
    # mounted rear-facing (yaw=π), matching puzzlebot_sim's laser_yaw=π and the
    # real robot.  RSP publishes /robot_description + base_footprint→base_link
    # (fixed) + base_link→{wheels,lidar_link,caster,lifter,camera,imu}.
    # puzzlebot_sim provides odom→base_footprint, so the chain
    # map(slam)→odom→base_footprint→base_link→links is a clean tree.
    robot_description = ParameterValue(
        Command(['xacro ', urdf, ' is_sim:=false is_ignition:=false']),
        value_type=str)

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': False}],
        output='screen')

    # Supplies /joint_states (wheels=continuous, lifter=prismatic) at 0 so RSP
    # can broadcast base_link→{wheel_*,lifter_*} TFs; reads /robot_description.
    jsp = Node(
        package='joint_state_publisher', executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': False}],
        output='screen')

    sim = Node(
        package='slam', executable='puzzlebot_sim', name='puzzlebot_sim',
        output='screen',
        parameters=[{
            'room_size_x': 4.86,
            'room_size_y': 3.76,
            # RPLidar A1M8 specs — the real unit sustains 10 Hz; tune SLAM at
            # the same cadence the robot will actually run so the certainty /
            # decay gates see the real per-second scan budget.
            'lidar_hz': 10.0,
            'lidar_beams': 360,
            'lidar_range_min': 0.15,
            'lidar_range_max': 12.0,
            'lidar_range_noise_pct': 0.01,
            'lidar_range_noise_floor': 0.005,
            'lidar_dropout_prob': 0.03,
            'lidar_latency': 0.15,
            'lidar_stamp_on_publish': True,
            # Puzzlebot wheel slip
            'slip_rotation': 0.12,
            'slip_translation': 0.04,
        }],
    )

    # puzzlebot_sim stamps scans on PUBLISH with a 0.15 s capture-to-delivery
    # latency (lidar_latency above), mimicking the real A1.  Feed slam_node the
    # matching scan_time_offset so the SAME latency-compensation path
    # (slam_node.cpp: scan_time_offset_) is exercised in sim as on the robot —
    # this catches double-wall regressions before they reach hardware.
    slam = TimerAction(period=1.0, actions=[Node(
        package='slam', executable='slam_node', name='slam_node',
        output='screen',
        parameters=[params, {'use_sim_time': False, 'scan_time_offset': 0.15}],
    )])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_cfg], output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([rviz_arg, rsp, jsp, sim, slam, rviz])
