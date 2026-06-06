"""
nav_node.py — Navigation node for the Puzzlebot AMR.

Loads a pre-saved map + waypoint file at startup (not from /map topic).
Accepts goal waypoint names on /goal_waypoint, runs A* immediately, then
follows the path with a pure-pursuit controller.

Obstacle avoidance: Bug1 algorithm.
  Phase 1 (WALL_FOLLOWING)   — right-hand wall follow; record leave point.
  Phase 2 (RETURNING_LEAVE)  — navigate to leave point; replan A* and resume.

Subscriptions
-------------
/slam_pose      (geometry_msgs/PoseStamped)  — pose from SLAM EKF (frame='map')
/goal_waypoint  (std_msgs/String)            — waypoint name, "" or "stop" to cancel
/scan           (sensor_msgs/LaserScan)      — obstacle detection / wall following

Publications
------------
/cmd_vel        (geometry_msgs/Twist)        — velocity commands
/plan           (nav_msgs/Path)              — planned path (frame='map')
/nav_status     (std_msgs/String)            — IDLE | PLANNING | FOLLOWING:<name>
                                               ARRIVED:<name> | ERROR:<reason>
                                               WALL_FOLLOWING:<name> | RETURNING_LEAVE:<name>
                                               WAITING_FOR_PATH:<name> (no route: holding+retrying)
"""

from __future__ import annotations

import math
import os
import struct
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                        qos_profile_sensor_data)

import tf2_ros
from rclpy.time import Time as RclpyTime

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from navigation.map_io import load_map, load_waypoints
from navigation.a_star import plan as astar_plan
from navigation.local_costmap import LocalCostmap

WorldPt = Tuple[float, float]


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

class _State:
    IDLE              = 'IDLE'
    FOLLOWING         = 'FOLLOWING'
    ALIGNING          = 'ALIGNING'          # in-place heading alignment at goal
    ARRIVED           = 'ARRIVED'
    WAITING_FOR_CLEAR = 'WAITING_FOR_CLEAR' # path blocked, waiting for dynamic obstacle
    WALL_FOLLOWING    = 'WALL_FOLLOWING'    # Bug1 phase 1
    RETURNING_LEAVE   = 'RETURNING_LEAVE'   # Bug1 phase 2
    WAITING_FOR_PATH  = 'WAITING_FOR_PATH'  # no route to goal: hold + retry plan


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class NavNode(Node):
    """Pure-pursuit + Bug1 obstacle avoidance navigation node."""

    def __init__(self) -> None:
        super().__init__('nav_node')

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter('map_yaml',            '~/ros2_maps/warehouse.yaml')
        self.declare_parameter('waypoints_yaml',      '~/ros2_maps/waypoints.yaml')
        self.declare_parameter('inflation_radius',    0.30)
        # Hard floor for A* fallback inflation.  Paths planned with less
        # inflation than this would put the robot literally inside or
        # next to walls — the fallback ladder will REFUSE to drop below
        # this value.  Default = robot physical radius (~10 cm) so we
        # never produce a path that the robot's body can't physically
        # clear.
        self.declare_parameter('inflation_radius_min', 0.10)
        self.declare_parameter('linear_speed',        0.15)
        self.declare_parameter('angular_kp',          2.0)
        # Derivative (damping) gain for the pure-pursuit heading loop.
        # The controller used to be pure-P, which — with omega clamped to
        # angular_max — saturated at ~6° of heading error and behaved like
        # bang-bang, exciting a limit cycle against the twist_relay accel
        # ramp (≈0.84 s to reverse omega).  angular_kd adds active damping
        # on the (filtered) heading-error rate so the loop has non-zero ζ.
        # angular_kd_tau is the low-pass time constant applied to that
        # derivative (blocks EKF pose jitter / lookahead segment-snap from
        # becoming omega chatter).  Both are ROS params so they can be
        # live-tuned with `ros2 param set` during a tuning run.
        self.declare_parameter('angular_kd',          0.5)
        self.declare_parameter('angular_kd_tau',      0.20)
        self.declare_parameter('angular_max',         1.2)
        self.declare_parameter('goal_tolerance',      0.20)
        self.declare_parameter('lookahead_distance',  0.40)
        self.declare_parameter('control_rate',        10.0)
        # Drive backwards when the lookahead point sits behind the
        # robot AND is close.  Lets the robot escape tight pockets
        # without sweeping its chassis through whatever forced the
        # tight turn.  reverse_max_dist caps how far we'll drive in
        # reverse before insisting on a forward maneuver.
        self.declare_parameter('allow_reverse',       True)
        self.declare_parameter('reverse_max_dist',    0.60)
        self.declare_parameter('obstacle_distance',   0.50)   # front-arc stop threshold (m)
        self.declare_parameter('obstacle_angle_deg',  55.0)   # half-angle of front arc (deg)
        self.declare_parameter('wall_follow_dist',    0.45)   # target wall distance (m)
        self.declare_parameter('heading_tolerance',   0.12)   # rad (~7°)
        # Minimum forward speed (m/s) for the final creep to a goal.  The
        # pure-pursuit speed scales with distance (min(1, dist/0.5)) and decays
        # to a few mm/s near the goal — below the motor deadband — so a TIGHT
        # goal_tolerance would never be reached (the robot stalls cm short).
        # When the robot is aimed at the goal and the scaled speed drops under
        # this floor, we command this instead so the last few cm close.  0
        # disables the floor.
        self.declare_parameter('approach_speed_min',  0.04)
        # Minimum in-place rotation speed (rad/s) during heading alignment at
        # the goal, so a TIGHT heading_tolerance is reachable instead of
        # stalling in the rotational deadband.  Was hard-coded 0.15.
        self.declare_parameter('align_speed_min',     0.15)
        # Robot body frame the scan angles must be expressed in.  The LiDAR
        # is mounted rotated relative to base_link (0 in sim, π rear-mount on
        # the real robot — see URDF lidar_yaw).  We resolve base_frame→scan
        # frame via TF on the first scan and add that yaw to every ray angle,
        # exactly like slam_node does.  Without it the front-arc, wall-follow
        # and local costmap point 180° the wrong way on the real robot.
        self.declare_parameter('base_frame',          'base_link')

        # Pose source for the steering loop.  'slam_pose' (default) steers on
        # the RAW /slam_pose heading — the unfiltered AMCL estimate, which
        # steps every scan on particle resampling, scan-to-map refine (±8°)
        # and loop closure.  Those steps feed the pure-pursuit P term and the
        # PD derivative directly, so the robot weaves chasing pose jitter.
        # Set 'tf' to instead read map→base_frame from the TF tree each tick:
        # the smooth (slam tf_alpha / relay-EMA) map→odom composed with the
        # 50 Hz wheel odom→base — a complementary filter that rejects the
        # per-scan jitter while still tracking real rotation between scans.
        # /slam_pose stays subscribed as the fallback (see _maybe_update_pose_
        # from_tf), so a TF lookup failure never stalls control.  OPT-IN:
        # confirm with the pose-jitter probe (and rule out scan latency)
        # before flipping it, and on the laptop also set the relay's
        # tf_smoothing_alpha < 1.0 (the relay rebuilds map→odom from raw
        # /slam_pose and otherwise passes the jitter straight through).
        self.declare_parameter('pose_source',         'slam_pose')

        # ── Local costmap (rolling LiDAR memory) ──────────────────────
        self.declare_parameter('local_costmap_radius',        2.0)
        self.declare_parameter('local_costmap_decay_tau',     0.7)
        self.declare_parameter('local_costmap_occ_threshold', 0.4)

        # ── Continuous path validity check ────────────────────────────
        # Lookahead is intentionally short: we only want to trigger a
        # replan if there's an obstacle IMMEDIATELY in front of the
        # robot's planned route, not anywhere within the next 3 m.
        # A wider lookahead caused replan flapping because distant
        # noise in the costmap would constantly flag the path as bad.
        self.declare_parameter('path_validity_lookahead', 1.0)
        self.declare_parameter('path_blocked_hysteresis', 12)
        # The validity check inflates obstacles by this factor of the
        # PLAN-TIME inflation when scanning the upcoming path.  Using a
        # value < 1.0 avoids flapping: A* places the path exactly at the
        # planner's inflation boundary, so a validity check at the SAME
        # inflation would always trigger and force a replan on every tick.
        # 0.5 means "replan only when something is half the planner
        # margin closer than expected" — robust against costmap noise.
        self.declare_parameter('path_validity_inflation_factor', 0.5)
        # Minimum seconds between two consecutive replans.  After a
        # replan, we COMMIT to the new path for at least this long even
        # if the validity check screams.  The safety bubble still halts
        # the robot for true collisions, so this cooldown is safe.
        self.declare_parameter('replan_cooldown', 3.0)
        # Master switch for validity-triggered replans during FOLLOWING.
        # When False (default), once a path is found the robot commits
        # to it until arrival OR a hard safety event (bubble → wait →
        # Bug1 fallback).  Set True if you want the robot to opportunis-
        # tically re-plan around obstacles its costmap discovers en route.
        self.declare_parameter('enable_validity_replan', False)

        # ── Runtime safety bubble (OFF by default) ────────────────────
        # The safety bubble is OFF by default because the user's intent
        # is "give me extra margin at PLAN time so the path itself stays
        # away from walls" — that's what `inflation_radius` does.  The
        # bubble fired on spurious LiDAR rays in the middle of an open
        # area, stopping the robot for no reason.  Set enable_safety_
        # bubble: true to re-enable as a runtime check against dynamic
        # obstacles (people walking in front, things rolling into the
        # path).  When enabled, bubble radius is clamped between
        # safety_bubble_min and safety_bubble_radius based on the path's
        # plan-time inflation.
        self.declare_parameter('enable_safety_bubble',     False)
        self.declare_parameter('safety_bubble_radius',     0.25)
        self.declare_parameter('safety_bubble_min',        0.10)
        # Hysteresis (ticks of consecutive breach before entering WAIT)
        # and min ray count for a tick to COUNT as breached.  Only
        # relevant when enable_safety_bubble: true.
        self.declare_parameter('bubble_breach_hysteresis', 5)
        self.declare_parameter('bubble_min_ray_count',     3)
        # When false (default) the robot does NOT fall back to Bug1 if the
        # WAIT-for-clear timeout expires.  In a fully-mapped environment
        # Bug1's wall-follow rarely helps (the planner already routed
        # around static obstacles) and frequently terminates with
        # "ERROR: Bug1 blocked". Set true if operating in maps with
        # unmodelled obstacles you want the robot to circumnavigate.
        self.declare_parameter('enable_bug1',          False)

        # ── Dynamic-obstacle wait policy ──────────────────────────────
        self.declare_parameter('wait_for_clear_timeout', 5.0)

        # ── Bug1 wall-follow watchdogs ────────────────────────────────
        # Absolute cap on a single circumnavigation, and a no-progress cap
        # (closest-to-goal distance must improve within this window).  Both
        # rescue the cases the legacy vicinity-gated 25 s timeout misses.
        self.declare_parameter('wall_follow_total_timeout',      90.0)
        self.declare_parameter('wall_follow_noprogress_timeout', 15.0)

        # ── FOLLOWING stuck watchdog ──────────────────────────────────
        # If distance-to-goal hasn't dropped by `follow_progress_eps` within
        # `follow_progress_timeout` s, the robot is wedged: emit ERROR so the
        # mission layer (SEARCH) can move on instead of hanging.
        self.declare_parameter('follow_progress_timeout', 12.0)
        self.declare_parameter('follow_progress_eps',     0.05)

        # ── No-path retry (hold-and-wait for a blocked route to open) ─
        # When the planner finds NO path at all (robot boxed in, a gate/door
        # sealing the only route, an obstacle the global plan can't get
        # around), DON'T give up: stop and re-plan every `no_path_retry_period`
        # s until a route appears (e.g. the door opens), then resume FOLLOWING.
        # Distinct from the FOLLOWING stuck watchdog above, which fires when a
        # path DOES exist but the robot can't progress on it (→ ERROR so SEARCH
        # skips).  `no_path_retry_timeout` <= 0 retries forever (the operator
        # can always abort via /goal_waypoint "stop"); > 0 gives up with ERROR
        # after that many seconds so a mission can move on.
        self.declare_parameter('no_path_retry',          True)
        self.declare_parameter('no_path_retry_period',    2.0)
        self.declare_parameter('no_path_retry_timeout',  -1.0)

        # ── System mode (gates goal acceptance) ───────────────────────
        # When mode != NAVIGATION, /goal_waypoint commands are rejected with
        # an ERROR status so we never try to navigate while the operator is
        # actively mapping.
        self.declare_parameter('start_mode', 'navigation')
        sm = str(self.get_parameter('start_mode').value).lower()
        if sm not in ('mapping', 'navigation'):
            sm = 'navigation'
        self._system_mode = 'NAVIGATION' if sm == 'navigation' else 'MAPPING'

        # Bug1 reachability guard: the wall-follow only runs after the robot
        # enters WAITING_FOR_CLEAR, which can only happen via the safety bubble
        # or validity-replan.  Flipping enable_bug1 alone is a silent no-op —
        # warn so the operator knows to enable a trigger too.
        if (bool(self.get_parameter('enable_bug1').value)
                and not bool(self.get_parameter('enable_safety_bubble').value)
                and not bool(self.get_parameter('enable_validity_replan').value)):
            self.get_logger().warn(
                'enable_bug1=true but both enable_safety_bubble and '
                'enable_validity_replan are false: nothing enters '
                'WAITING_FOR_CLEAR, so Bug1 wall-follow can NEVER trigger. '
                'Enable one of those to use Bug1.'
            )

        map_yaml       = os.path.expanduser(self.get_parameter('map_yaml').value)
        waypoints_yaml = os.path.expanduser(self.get_parameter('waypoints_yaml').value)

        # ── Load map ────────────────────────────────────────────────────
        self.get_logger().info(f'Loading map from {map_yaml}')
        try:
            self._grid, self._origin_x, self._origin_y, self._resolution = \
                load_map(map_yaml)
            self.get_logger().info(
                f'Map loaded: {self._grid.shape[1]}x{self._grid.shape[0]} cells, '
                f'res={self._resolution} m/cell, '
                f'origin=({self._origin_x}, {self._origin_y})'
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to load map: {exc}')
            self._grid = None
            self._origin_x = -10.0
            self._origin_y = -10.0
            self._resolution = 0.05

        # ── Load waypoints ─────────────────────────────────────────────
        self._waypoints_path = waypoints_yaml
        self.get_logger().info(f'Loading waypoints from {waypoints_yaml}')
        try:
            self._waypoints = load_waypoints(waypoints_yaml)
            self.get_logger().info(
                f'Waypoints loaded: {list(self._waypoints.keys())}'
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to load waypoints: {exc}')
            self._waypoints = {}
        # File mtime at load, so the background watcher (below) can auto-reload
        # when the file changes — dashboard, waypoint_recorder, or a manual edit
        # — WITHOUT relaunching nav. 0.0 = missing/unreadable.
        self._waypoints_mtime = self._wp_file_mtime()

        # ── Navigation state ───────────────────────────────────────────
        self._pose_x: Optional[float] = None
        self._pose_y: Optional[float] = None
        self._pose_theta: Optional[float] = None
        self._latest_scan: Optional[LaserScan] = None

        # ── LiDAR mounting yaw (base_frame → scan frame) ────────────────
        # Resolved from TF on the first scan whose frame_id we can look up.
        # 0.0 until resolved (sim-safe); becomes ~π on the real rear-mounted
        # RPLidar A1.  Applied to every ray bearing before it is used for
        # obstacle detection, wall-following and costmap projection.
        self._base_frame: str = str(self.get_parameter('base_frame').value)
        self._lidar_yaw: float = 0.0
        self._lidar_yaw_resolved: bool = False
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._path: List[WorldPt] = []
        self._state: str = _State.IDLE
        self._current_goal_name: str = ''
        self._arrived_timer: Optional[object] = None

        # ── Pure-pursuit PD steering state ─────────────────────────────
        # The derivative term operates on the CONTINUOUS heading error and
        # is low-pass filtered.  It must be reset (via _reset_pd) whenever a
        # fresh path/segment context begins — a wholesale self._path swap
        # makes the lookahead point, and thus heading_error, step
        # discontinuously, and differentiating across that step would spike
        # omega.  _pd_valid stays False until we hold a trustworthy previous
        # sample so the first tick after a reset contributes no derivative.
        self._prev_heading_error: float = 0.0
        self._d_filt: float = 0.0
        self._pd_valid: bool = False

        # Bug1 state
        self._q_hit: Optional[WorldPt] = None    # where we first hit the obstacle
        self._q_leave: Optional[WorldPt] = None  # closest-to-goal point found so far
        self._d_leave: float = math.inf           # distance from q_leave to goal
        self._wf_goal: Optional[WorldPt] = None  # goal coordinates when Bug1 started
        self._wf_left_vicinity: bool = False      # True once we've moved away from q_hit
        self._wf_start_time: float = 0.0          # monotonic time when wall-follow started
        # Wall-follow watchdog (re-armed in _enter_wall_follow): bound the
        # circumnavigation by absolute time AND by genuine progress toward the
        # goal, so a robot circling a convex corner or spinning in place can't
        # wall-follow forever (the 25 s vicinity timeout only guards the phase
        # BEFORE the robot leaves the hit point).
        self._wf_best_d_to_goal: float = math.inf
        self._wf_last_progress_time: float = 0.0

        # FOLLOWING no-progress watchdog (re-armed by _reset_pd on every fresh
        # FOLLOWING context).  If the robot stops shrinking its distance-to-goal
        # it is wedged (clipping inflation, pose-jitter oscillation, a path the
        # costmap silently blocks); we then emit an ERROR so a waiting
        # navigate() poller (e.g. SEARCH between rollers) advances instead of
        # blocking on a goal that never ARRIVES.
        self._follow_best_dist: float = math.inf
        self._follow_progress_t: float = 0.0

        # WAITING_FOR_PATH bookkeeping (no route to goal → hold + retry plan)
        self._waitpath_start_time: float = 0.0
        self._waitpath_last_try: float = 0.0
        self._waitpath_last_log: float = 0.0

        # ── Local costmap (rolling LiDAR memory) ───────────────────────
        self._costmap = LocalCostmap(
            radius_m=self.get_parameter('local_costmap_radius').value,
            resolution=self._resolution if self._resolution else 0.05,
            decay_tau=self.get_parameter('local_costmap_decay_tau').value,
            occ_threshold=self.get_parameter('local_costmap_occ_threshold').value,
        )
        self._last_costmap_tick_time: Optional[float] = None

        # Inflation that produced the currently-followed path.  Used by
        # the path-validity check so its inflation is consistent with the
        # plan-time inflation (otherwise validity flaps and replan loops).
        self._path_plan_infl: float = float(
            self.get_parameter('inflation_radius').value
        )

        # Wall-clock (sec) of the last successful (re)plan.  Used to
        # gate validity-triggered replans so we don't thrash.
        self._last_replan_time: Optional[float] = None

        # Safety-bubble debouncing: count consecutive ticks the bubble
        # has been violated while FOLLOWING.  We only enter WAIT after
        # this exceeds `bubble_breach_hysteresis`.
        self._bubble_breach_ticks: int = 0

        # Path-validity hysteresis counter (consecutive blocked ticks)
        self._path_blocked_ticks: int = 0

        # WAITING_FOR_CLEAR bookkeeping
        self._wait_start_time: float = 0.0
        self._wait_last_replan_try: float = 0.0

        # ── QoS ────────────────────────────────────────────────────────
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ── Subscriptions ──────────────────────────────────────────────
        self.create_subscription(PoseStamped, '/slam_pose',     self._pose_cb, reliable_qos)
        self.create_subscription(String,      '/goal_waypoint', self._goal_cb, reliable_qos)
        self.create_subscription(LaserScan,   '/scan',          self._scan_cb, qos_profile_sensor_data)

        # /map (from slam_node): keep our planning grid in sync with the live
        # map. This means a saved-then-switched-to-NAV flow doesn't require a
        # launch restart — and a fresh cold boot with no map_yaml will still
        # work once SLAM publishes the first grid.
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, 1)

        # /system_mode (latched, transient_local) — dashboard publishes mode
        mode_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.create_subscription(
            String, '/system_mode', self._system_mode_cb, mode_qos,
        )

        # Reload waypoints from disk (called by the dashboard after it
        # writes a new waypoint to waypoints.yaml).
        self.create_service(Trigger, '~/reload_waypoints', self._reload_waypoints_cb)

        # Auto-reload: poll the waypoints file mtime at 1 Hz and reload when it
        # changes, so waypoints added by the dashboard, the waypoint_recorder, or
        # a manual edit become addressable WITHOUT relaunching nav (the explicit
        # ~/reload_waypoints service stays as a belt-and-suspenders trigger).
        self.create_timer(1.0, self._watch_waypoints)

        # ── Publishers ─────────────────────────────────────────────────
        self._cmd_vel_pub  = self.create_publisher(Twist,       '/cmd_vel_in',       10)
        self._plan_pub     = self.create_publisher(Path,        '/plan',             10)
        self._status_pub   = self.create_publisher(String,      '/nav_status',       10)
        self._walls_pub    = self.create_publisher(PointCloud2, '/map_walls',         1)
        self._markers_pub  = self.create_publisher(MarkerArray, '/waypoint_markers',  1)

        self._publish_map_walls()
        self._publish_waypoint_markers()
        self.create_timer(5.0, self._publish_map_walls)
        self.create_timer(5.0, self._publish_waypoint_markers)

        rate = self.get_parameter('control_rate').value
        self.create_timer(1.0 / rate, self._control_loop)

        self._publish_status('IDLE')
        self.get_logger().info('NavNode ready.')

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _scan_cb(self, msg: LaserScan) -> None:
        self._latest_scan = msg
        self._resolve_lidar_yaw(msg.header.frame_id)
        # Feed the costmap immediately so the next control tick sees fresh data.
        if self._pose_x is not None and self._costmap is not None:
            self._costmap.recenter(self._pose_x, self._pose_y)
            self._costmap.integrate_scan(
                msg, self._pose_x, self._pose_y, self._pose_theta,
                lidar_yaw=self._lidar_yaw,
            )

    def _resolve_lidar_yaw(self, scan_frame: str) -> None:
        """Cache the base_frame→scan_frame yaw from TF (once).

        Mirrors slam_node's resolveLaser: the RPLidar is mounted rotated
        relative to base_link (0 in sim, π on the real robot).  Every ray
        bearing is reported in the scan frame, so we add this yaw before
        treating it as a base_link-relative angle.
        """
        if self._lidar_yaw_resolved or not scan_frame:
            return
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, scan_frame, RclpyTime())
        except Exception:
            return  # TF not available yet — keep yaw=0.0, retry next scan
        q = tf.transform.rotation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._lidar_yaw = math.atan2(siny, cosy)
        self._lidar_yaw_resolved = True
        self.get_logger().info(
            f'LiDAR mount yaw ({self._base_frame}→{scan_frame}): '
            f'{math.degrees(self._lidar_yaw):.1f}°'
        )

    def _map_cb(self, msg: OccupancyGrid) -> None:
        """Replace the static planning grid with the latest map from SLAM.

        slam_node publishes /map on every Nth scan during MAPPING and once
        per save during NAVIGATION (latched data, but default QoS here is
        fine since slam re-publishes shortly after we connect).
        """
        w = msg.info.width
        h = msg.info.height
        if w == 0 or h == 0:
            return
        try:
            grid = np.asarray(msg.data, dtype=np.int8).reshape((h, w))
        except ValueError as exc:
            self.get_logger().warn(f'/map reshape failed ({w}x{h}): {exc}')
            return

        first_load = self._grid is None
        self._grid       = grid
        self._origin_x   = float(msg.info.origin.position.x)
        self._origin_y   = float(msg.info.origin.position.y)
        self._resolution = float(msg.info.resolution)

        if first_load:
            self.get_logger().info(
                f'/map received: {w}x{h} cells, res={self._resolution} m, '
                f'origin=({self._origin_x:.3f}, {self._origin_y:.3f})'
            )
            self._publish_map_walls()

    def _reload_waypoints_cb(self, request, response):
        """Re-read waypoints.yaml from disk so newly-added waypoints become
        addressable without restarting the node.
        """
        try:
            path = os.path.expanduser(self.get_parameter('waypoints_yaml').value)
            self._waypoints = load_waypoints(path)
            self._waypoints_mtime = self._wp_file_mtime()   # sync so the watcher won't re-fire
            self.get_logger().info(
                f'Waypoints reloaded: {list(self._waypoints.keys())}'
            )
            self._publish_waypoint_markers()
            response.success = True
            response.message = f'{len(self._waypoints)} waypoints loaded'
        except Exception as exc:
            response.success = False
            response.message = f'reload failed: {exc}'
        return response

    def _wp_file_mtime(self) -> float:
        """mtime of the waypoints file, or 0.0 if missing/unreadable."""
        try:
            return os.path.getmtime(self._waypoints_path)
        except OSError:
            return 0.0

    def _watch_waypoints(self) -> None:
        """1 Hz: reload waypoints if the file changed on disk — no relaunch.
        On a read failure (e.g. caught mid-write) it leaves the mtime unchanged
        so it retries next tick instead of going stale on a partial file."""
        mtime = self._wp_file_mtime()
        if mtime == 0.0 or mtime == self._waypoints_mtime:
            return
        try:
            wps = load_waypoints(self._waypoints_path)
        except Exception as exc:
            self.get_logger().warn(f'Waypoint auto-reload failed (will retry): {exc}')
            return
        self._waypoints = wps
        self._waypoints_mtime = mtime
        self.get_logger().info(
            f'Waypoints auto-reloaded ({len(wps)}): {list(wps.keys())}'
        )
        self._publish_waypoint_markers()

    def _system_mode_cb(self, msg: String) -> None:
        """Track the current system mode; cancel navigation if we leave it."""
        mode = msg.data.strip().upper()
        if mode not in ('MAPPING', 'NAVIGATION'):
            return
        if mode == self._system_mode:
            return
        prev = self._system_mode
        self._system_mode = mode
        self.get_logger().info(f'system_mode: {prev} → {mode}')
        if mode != 'NAVIGATION' and self._state not in (
            _State.IDLE, _State.ARRIVED
        ):
            self.get_logger().info('Cancelling active navigation due to mode change.')
            self._cancel_navigation()

    def _pose_cb(self, msg: PoseStamped) -> None:
        pos = msg.pose.position
        ori = msg.pose.orientation
        self._pose_x = pos.x
        self._pose_y = pos.y
        siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        self._pose_theta = math.atan2(siny_cosp, cosy_cosp)

    def _maybe_update_pose_from_tf(self) -> None:
        """When pose_source == 'tf', override the pose with a smooth TF read.

        Steers on map→base_frame from the TF tree instead of the raw
        /slam_pose heading.  That transform is the low-pass-smoothed map→odom
        (slam_node's tf_alpha on-board, or the map_odom_relay EMA on the
        laptop) composed with the 50 Hz wheel odom→base — i.e. a complementary
        filter: it rejects the per-scan AMCL / refine / loop-closure heading
        steps yet still tracks real rotation between scans, so it adds almost
        no lag to the loop (unlike low-passing /slam_pose wholesale).

        /slam_pose remains the fallback: any lookup failure (TF not yet
        published, time extrapolation, clock skew) returns early and leaves
        the last /slam_pose-derived pose in place, so the control loop never
        stalls or coasts on a stale exception path.
        """
        if str(self.get_parameter('pose_source').value).lower() != 'tf':
            return
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', self._base_frame, RclpyTime())
        except Exception:
            return  # keep last /slam_pose pose — never stall the control loop
        t = tf.transform.translation
        q = tf.transform.rotation
        self._pose_x = t.x
        self._pose_y = t.y
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._pose_theta = math.atan2(siny, cosy)

    def _goal_cb(self, msg: String) -> None:
        name = msg.data.strip()

        if name in ('', 'stop'):
            self._cancel_navigation()
            return

        # Reject goals while system is in MAPPING mode — we don't want the
        # robot driving around autonomously while the operator is building
        # a map (could collide with un-mapped obstacles or interfere with SLAM).
        if self._system_mode != 'NAVIGATION':
            self.get_logger().warn(
                f'Goal "{name}" rejected: system_mode={self._system_mode}, '
                'expected NAVIGATION.'
            )
            self._publish_status(f'ERROR: not in navigation mode')
            return

        if name not in self._waypoints:
            self.get_logger().error(f'Unknown waypoint: "{name}"')
            self._publish_status(f'ERROR: unknown waypoint {name}')
            return

        wp = self._waypoints[name]
        goal_x, goal_y = wp['x'], wp['y']

        self.get_logger().info(f'Goal received: "{name}" → ({goal_x:.2f}, {goal_y:.2f})')

        # A new goal supersedes any pending post-arrival → IDLE timer from the
        # PREVIOUS goal. Without this, arriving at goal A then receiving goal B
        # within the 1 s arrived-timeout window lets A's stale timeout fire while
        # we're FOLLOWING B — yanking us back to IDLE mid-route (path cleared,
        # robot coasts on the smoother's last command into a crash).
        self._cancel_arrived_timer()

        if self._pose_x is None:
            self.get_logger().error('No pose available yet — cannot plan.')
            self._publish_status('ERROR: no pose')
            return

        if self._grid is None:
            self.get_logger().error('No map loaded — cannot plan.')
            self._publish_status('ERROR: no map')
            return

        self._publish_status('PLANNING')
        # Plan over the STATIC map only — without the local LiDAR costmap
        # overlay.  The costmap accumulates transient hits within a 2 m
        # radius and frequently adds phantom obstacles (glancing wall
        # rays, dust, etc.) that force A* into long detours.  We want
        # the SHORTEST geometrically-valid path at goal time.  If a real
        # dynamic obstacle is in the way, the safety bubble triggers
        # WAITING_FOR_CLEAR, which calls _replan_from_current — and that
        # path DOES include the live costmap so it goes around the
        # blocker.  Best of both worlds.
        planning_grid = self._grid if self._grid is not None else None
        path = self._plan_with_fallback(
            planning_grid,
            self._pose_x, self._pose_y, goal_x, goal_y,
            label=f'"{name}"',
        )
        if path is None:
            # No route from here to the goal (boxed in / sealed passage).  Hold
            # and keep retrying instead of failing the goal, so the robot
            # resumes the moment the way opens (e.g. a door/gate clears).
            self._current_goal_name = name
            self._reset_pd()
            self._enter_waiting_for_path('initial plan found no route')
            return

        self.get_logger().info(f'Path found: {len(path)} waypoints to "{name}"')
        self._path = path
        self._current_goal_name = name
        self._path_blocked_ticks = 0
        self._last_replan_time = self.get_clock().now().nanoseconds * 1e-9
        self._reset_pd()
        self._state = _State.FOLLOWING
        self._publish_status(f'FOLLOWING: {name}')

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """Dispatches to per-state controllers at control_rate Hz."""
        self._publish_plan()

        if self._pose_x is None:
            return

        # Optionally replace the raw /slam_pose pose with the smooth,
        # odom-propagated TF reading (pose_source == 'tf').  Done here so the
        # costmap and every per-state controller below steer on the same
        # de-jittered pose.  No-op (and zero cost) when pose_source is the
        # default 'slam_pose'.
        self._maybe_update_pose_from_tf()

        # Tick the local costmap (decay + recenter).  This runs in every
        # state so the costmap stays fresh even while waiting or wall-following.
        self._tick_costmap()

        px, py, theta = self._pose_x, self._pose_y, self._pose_theta

        # ── 360° safety bubble: emergency stop ONLY in FOLLOWING ──
        # Skipped entirely when enable_safety_bubble is False (default):
        # the user prefers the safety margin to come from the planner's
        # `inflation_radius`, not from runtime LiDAR-vs-path checks that
        # over-fire on noise.
        if (self._state == _State.FOLLOWING
                and bool(self.get_parameter('enable_safety_bubble').value)):
            if self._safety_bubble_violated():
                self._bubble_breach_ticks += 1
                hyst = int(self.get_parameter('bubble_breach_hysteresis').value)
                if self._bubble_breach_ticks >= hyst:
                    self._publish_stop()
                    self.get_logger().warn(
                        f'Safety bubble breached for {self._bubble_breach_ticks} '
                        'ticks — entering WAITING_FOR_CLEAR'
                    )
                    self._bubble_breach_ticks = 0
                    self._enter_waiting(px, py)
                    return
                # Below the hysteresis threshold: stop motion this tick
                # but keep the FOLLOWING state so we resume next tick if
                # the breach clears.
                self._publish_stop()
                return
            else:
                # Bubble clear — reset the counter and continue with the
                # normal FOLLOWING control logic below.
                self._bubble_breach_ticks = 0

        if self._state == _State.WAITING_FOR_CLEAR:
            self._waiting_update(px, py)
            return

        if self._state == _State.WAITING_FOR_PATH:
            self._waiting_for_path_update(px, py)
            return

        if self._state == _State.WALL_FOLLOWING:
            self._wall_follow_update(px, py)
            return

        if self._state == _State.ALIGNING:
            self._aligning_update(theta)
            return

        if self._state == _State.RETURNING_LEAVE:
            self._returning_leave_update(px, py, theta)
            return

        if self._state == _State.IDLE:
            return  # yield control to teleop / other nodes
        if self._state != _State.FOLLOWING:
            self._publish_stop()
            return

        # ── FOLLOWING ────────────────────────────────────────────────
        if not self._path:
            self._on_arrived()
            return

        fx, fy = self._path[-1]
        dist_to_goal = math.sqrt((fx - px) ** 2 + (fy - py) ** 2)

        goal_tol = self.get_parameter('goal_tolerance').value
        if dist_to_goal < goal_tol:
            self._on_arrived()
            return

        # ── FOLLOWING no-progress watchdog ─────────────────────────────
        # Track the best (closest) distance-to-goal; if it doesn't improve by
        # at least follow_progress_eps within follow_progress_timeout seconds
        # the robot is wedged and will never ARRIVE.  Emit an ERROR status
        # (NOT a silent IDLE): a polling navigate() returns 'error' on an ERROR
        # prefix, so SEARCH advances to the next roller instead of blocking on
        # a goal it can't reach.  The watchdog is re-armed by _reset_pd on every
        # fresh path (new goal, replan, WAIT→resume), so legitimate replans
        # grant a fresh window and only a true stall trips it.
        now_s = self.get_clock().now().nanoseconds * 1e-9
        eps = float(self.get_parameter('follow_progress_eps').value)
        if dist_to_goal < self._follow_best_dist - eps:
            self._follow_best_dist = dist_to_goal
            self._follow_progress_t = now_s
        elif now_s - self._follow_progress_t > float(
                self.get_parameter('follow_progress_timeout').value):
            self.get_logger().error(
                f'FOLLOWING made no progress toward "{self._current_goal_name}" '
                f'for {self.get_parameter("follow_progress_timeout").value:.0f} s '
                '— declaring stuck.'
            )
            self._publish_stop()
            self._cancel_arrived_timer()
            self._path = []
            self._state = _State.IDLE
            self._publish_status(f'ERROR: stuck {self._current_goal_name}')
            return

        # Continuous path validity check.  Default behaviour (with
        # enable_validity_replan=False) is to COMMIT to the path the
        # planner produced and never swap it out from under the
        # controller — the user explicitly asked for "stay with the
        # first path you find".  The safety bubble + WAITING_FOR_CLEAR
        # + Bug1 fallback handle real obstacles.
        if bool(self.get_parameter('enable_validity_replan').value):
            now_s = self.get_clock().now().nanoseconds * 1e-9
            cooldown = float(self.get_parameter('replan_cooldown').value)
            in_cooldown = (
                self._last_replan_time is not None
                and (now_s - self._last_replan_time) < cooldown
            )

            if in_cooldown:
                self._path_blocked_ticks = 0
            else:
                if self._path_ahead_blocked(px, py):
                    self._path_blocked_ticks += 1
                else:
                    self._path_blocked_ticks = 0

                hysteresis = int(self.get_parameter('path_blocked_hysteresis').value)
                if self._path_blocked_ticks >= hysteresis:
                    self._publish_stop()
                    self.get_logger().info(
                        f'Path blocked for {self._path_blocked_ticks} ticks — replanning'
                    )
                    self._path_blocked_ticks = 0
                    if self._replan_from_current(px, py):
                        return
                    self.get_logger().warn(
                        'Replan failed — entering WAITING_FOR_CLEAR (dynamic-obstacle wait)'
                    )
                    self._enter_waiting(px, py)
                    return
        else:
            # Keep the counter sane in case it's used elsewhere.
            self._path_blocked_ticks = 0

        lookahead = self.get_parameter('lookahead_distance').value
        if dist_to_goal < lookahead:
            lx, ly = fx, fy
        else:
            lx, ly = self._find_lookahead(px, py, lookahead)

        heading_error = self._wrap_angle(math.atan2(ly - py, lx - px) - theta)

        kp     = self.get_parameter('angular_kp').value
        kd     = self.get_parameter('angular_kd').value
        kd_tau = self.get_parameter('angular_kd_tau').value
        wmax   = self.get_parameter('angular_max').value
        vmax   = self.get_parameter('linear_speed').value

        # ── Reverse maneuver decision ──────────────────────────────
        # When the lookahead point is in the rear hemisphere AND close
        # to the robot, drive backwards instead of pirouetting 180°.
        # This handles the case where the robot is wedged near a wall
        # and the next leg of the plan curls back behind it.
        # Threshold: |heading_error| > π/2 + a small dead-band, and
        # lookahead dist below `reverse_max_dist` so we never drive
        # backwards across long open stretches (less precise control).
        allow_reverse    = self.get_parameter('allow_reverse').value
        reverse_max_dist = self.get_parameter('reverse_max_dist').value
        lookahead_dist   = math.hypot(lx - px, ly - py)
        use_reverse = (
            allow_reverse
            and abs(heading_error) > (math.pi / 2.0 + 0.1)
            and lookahead_dist <= reverse_max_dist
        )

        if use_reverse:
            ctrl_error = self._wrap_angle(heading_error - math.pi)
            sign = -1.0
        else:
            ctrl_error = heading_error
            sign = +1.0

        # ── Filtered derivative (damping) term ─────────────────────
        # Computed on the CONTINUOUS heading_error, NOT on ctrl_error.
        # A reverse maneuver only flips ctrl_error (by π) and the linear-
        # speed sign — heading_error itself does not jump — so its
        # derivative is spike-free across a reverse toggle and equals
        # d(ctrl_error)/dt in both regimes.  That means the damping term
        # is applied with the SAME sign whether driving forward or in
        # reverse (no `sign` multiplier — that would be anti-damping in
        # reverse).  The wrap on the difference handles the heading
        # crossing ±π between ticks; the low-pass (tau = angular_kd_tau)
        # stops EKF pose jitter and lookahead segment-snap from turning
        # into omega chatter.
        dt = 1.0 / self.get_parameter('control_rate').value
        if self._pd_valid and dt > 1e-6:
            raw_de = self._wrap_angle(heading_error - self._prev_heading_error) / dt
            # Belt-and-suspenders clamp against any residual step the
            # path-reset gates miss (segment snap, single pose glitch):
            # cap the raw rate at what the actuator could use anyway.
            raw_de = self._clamp(raw_de, -2.0, 2.0)
            alpha = dt / (kd_tau + dt)
            self._d_filt += alpha * (raw_de - self._d_filt)
        else:
            # First tick after a (re)plan / FOLLOWING re-entry: no
            # trustworthy previous sample yet, so contribute no derivative.
            self._d_filt = 0.0
        self._prev_heading_error = heading_error
        self._pd_valid = True

        omega = self._clamp(kp * ctrl_error + kd * self._d_filt, -wmax, wmax)
        speed_mag = vmax * max(0.0, math.cos(ctrl_error)) * min(1.0, dist_to_goal / 0.5)

        # ── Final-approach speed floor ─────────────────────────────────
        # The distance term (min(1, dist/0.5)) decays speed_mag to a few mm/s
        # just outside goal_tolerance — under the motor deadband, so a tight
        # tolerance would stall cm short and never trigger arrival.  When the
        # robot is still outside tolerance (arrival is checked at the top of
        # the loop) AND pointed at the goal (so the floor drives it TOWARD the
        # goal, not sideways into a wall), raise the magnitude to
        # approach_speed_min.  Applies symmetrically to a reverse overshoot-
        # correction: `sign` carries the direction and the small |ctrl_error|
        # gate holds in both regimes (ctrl_error ≈ 0 when backing straight in).
        approach_min = self.get_parameter('approach_speed_min').value
        if (approach_min > 0.0
                and abs(ctrl_error) < math.radians(20.0)
                and 0.0 < speed_mag < approach_min):
            speed_mag = approach_min

        speed = sign * speed_mag

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = omega
        self._cmd_vel_pub.publish(cmd)

    # ------------------------------------------------------------------
    # WAITING_FOR_CLEAR — pause for a dynamic obstacle before falling back to Bug1
    # ------------------------------------------------------------------

    def _enter_waiting(self, px: float, py: float) -> None:
        """Stop the robot and start the wait-for-clear timer."""
        self._publish_stop()
        self._wait_start_time = self.get_clock().now().nanoseconds * 1e-9
        self._wait_last_replan_try = 0.0
        self._path_blocked_ticks = 0
        self._state = _State.WAITING_FOR_CLEAR
        self._publish_status(f'WAITING_FOR_CLEAR: {self._current_goal_name}')
        self.get_logger().info(
            f'Pausing up to {self.get_parameter("wait_for_clear_timeout").value:.1f} s, '
            'attempting replan around obstacle.'
        )

    def _waiting_update(self, px: float, py: float) -> None:
        """While waiting: prefer the cheapest recovery in this order:
          1) bubble cleared → resume ORIGINAL path (no replan)
          2) try a throttled replan around the obstacle
          3) on timeout, if bubble is clear, resume original path
             (the breach was likely transient)
          4) on timeout with bubble still breached, give up (or Bug1).
        """
        self._publish_stop()
        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self._wait_start_time
        timeout = self.get_parameter('wait_for_clear_timeout').value
        bubble_clear = not self._safety_bubble_violated()

        # 1) Fastest recovery: bubble cleared and we still have a path.
        #    Resume FOLLOWING the original path — no replan needed.
        if bubble_clear and self._path:
            self.get_logger().info(
                'Bubble cleared — resuming original path.'
            )
            self._bubble_breach_ticks = 0
            self._reset_pd()
            self._state = _State.FOLLOWING
            self._publish_status(f'FOLLOWING: {self._current_goal_name}')
            return

        # 2) Throttled replan around the obstacle (uses live costmap).
        replan_period = 1.0
        if now - self._wait_last_replan_try >= replan_period:
            self._wait_last_replan_try = now
            if self._replan_from_current(px, py):
                self.get_logger().info(
                    'Replan around obstacle succeeded — resuming FOLLOWING.'
                )
                return

        # 3 / 4) Timeout handling.
        if elapsed >= timeout:
            # If the bubble cleared but no path exists, we can't continue.
            if bubble_clear and self._path:
                self.get_logger().info(
                    f'Wait-for-clear timeout but bubble now clear — '
                    'resuming original path.'
                )
                self._bubble_breach_ticks = 0
                self._reset_pd()
                self._state = _State.FOLLOWING
                self._publish_status(f'FOLLOWING: {self._current_goal_name}')
                return
            if bool(self.get_parameter('enable_bug1').value):
                self.get_logger().warn(
                    f'Wait-for-clear timeout ({timeout:.1f} s) — entering Bug1 wall follow.'
                )
                self._enter_wall_follow(px, py)
            else:
                # Obstacle won't clear and no detour exists: hold for the route
                # to open (door/gate) rather than abandoning the goal.
                self._enter_waiting_for_path(
                    f'wait-for-clear timeout ({timeout:.1f} s), Bug1 disabled'
                )

    # ------------------------------------------------------------------
    # WAITING_FOR_PATH — no route to the goal: hold position and retry
    # ------------------------------------------------------------------

    def _enter_waiting_for_path(self, reason: str) -> None:
        """No route to the goal right now (planner / Bug1 found none): stop and
        keep re-planning until one opens — e.g. a door/gate clears — instead of
        giving up.  All the "no path" dead-ends funnel here so a boxed-in robot
        waits for the way to open rather than aborting the goal.

        Disabled (no_path_retry=false) restores the old behaviour: stop, emit
        ERROR and drop to IDLE so the caller fails fast.
        """
        self._publish_stop()
        self._path = []
        if not bool(self.get_parameter('no_path_retry').value):
            self._state = _State.IDLE
            self._publish_status(f'ERROR: no path to {self._current_goal_name}')
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._state != _State.WAITING_FOR_PATH:
            self._waitpath_start_time = now
            self._waitpath_last_log = now
            self.get_logger().warn(
                f'No path to "{self._current_goal_name}" ({reason}) — holding and '
                f'retrying every {self.get_parameter("no_path_retry_period").value:.1f} s '
                'until the route opens.'
            )
        self._waitpath_last_try = now
        self._state = _State.WAITING_FOR_PATH
        self._publish_status(f'WAITING_FOR_PATH: {self._current_goal_name}')

    def _waiting_for_path_update(self, px: float, py: float) -> None:
        """While holding for a path: stay stopped and re-plan (live-scan grid,
        so a cleared dynamic obstacle is seen) every no_path_retry_period.  On
        success _replan_from_current transitions to FOLLOWING.  With a positive
        no_path_retry_timeout, give up to ERROR/IDLE after that long so a
        mission isn't pinned forever on an unreachable goal."""
        self._publish_stop()
        now = self.get_clock().now().nanoseconds * 1e-9

        timeout = float(self.get_parameter('no_path_retry_timeout').value)
        if timeout > 0.0 and (now - self._waitpath_start_time) >= timeout:
            self.get_logger().error(
                f'Still no path to "{self._current_goal_name}" after {timeout:.0f} s '
                '— giving up.'
            )
            self._publish_stop()
            self._path = []
            self._state = _State.IDLE
            self._publish_status(f'ERROR: no path to {self._current_goal_name}')
            return

        # Heartbeat so the operator sees it is still trying (not hung).
        if now - self._waitpath_last_log >= 10.0:
            self._waitpath_last_log = now
            waited = now - self._waitpath_start_time
            self.get_logger().info(
                f'WAITING_FOR_PATH: still no route to "{self._current_goal_name}" '
                f'after {waited:.0f} s — retrying.'
            )

        period = float(self.get_parameter('no_path_retry_period').value)
        if now - self._waitpath_last_try < period:
            return
        self._waitpath_last_try = now
        if self._replan_from_current(px, py):
            self.get_logger().info(
                f'Route to "{self._current_goal_name}" opened — resuming FOLLOWING.'
            )
        # else: still blocked — stay in WAITING_FOR_PATH (status already set).

    # ------------------------------------------------------------------
    # Bug1 — wall following (phase 1)
    # ------------------------------------------------------------------

    def _enter_wall_follow(self, px: float, py: float) -> None:
        """Start Bug1: record hit point, begin circumnavigation."""
        wp = self._waypoints.get(self._current_goal_name)
        if wp is None:
            self._state = _State.IDLE
            return
        gx, gy = wp['x'], wp['y']

        self._q_hit = (px, py)
        self._q_leave = (px, py)
        self._d_leave = math.sqrt((gx - px) ** 2 + (gy - py) ** 2)
        self._wf_goal = (gx, gy)
        self._wf_left_vicinity = False
        self._wf_start_time = self.get_clock().now().nanoseconds * 1e-9
        # Arm the wall-follow watchdogs: best-so-far distance to goal (for the
        # no-progress cap) and the absolute start time (already in _wf_start_time).
        self._wf_best_d_to_goal = self._d_leave
        self._wf_last_progress_time = self._wf_start_time
        self._state = _State.WALL_FOLLOWING
        self._publish_status(f'WALL_FOLLOWING: {self._current_goal_name}')
        self.get_logger().info(
            f'Bug1: hit obstacle at ({px:.2f}, {py:.2f}), starting circumnavigation'
        )

    def _wall_follow_update(self, px: float, py: float) -> None:
        """Bug1 phase 1: right-hand wall follow; track leave point."""
        scan = self._latest_scan
        if scan is None:
            self._publish_stop()
            return

        # Timeout: if stuck near hit point for >25 s, try replanning and give up.
        elapsed = self.get_clock().now().nanoseconds * 1e-9 - self._wf_start_time
        if elapsed > 25.0 and not self._wf_left_vicinity:
            self.get_logger().warn(
                'Bug1: stuck near hit point for 25 s — attempting live-scan replan'
            )
            if not self._replan_from_current(px, py):
                self._enter_waiting_for_path('Bug1 stuck at hit point')
            return

        # Absolute cap regardless of hit-vicinity.  The 25 s guard above only
        # fires while _wf_left_vicinity is False, and that flag latches True
        # forever once the robot moves >0.5 m from the hit point — after which
        # NO timeout could fire.  Without this an endless convex-corner orbit or
        # in-place spin would wall-follow indefinitely.
        if elapsed > float(self.get_parameter('wall_follow_total_timeout').value):
            self.get_logger().error(
                'Bug1: wall-follow exceeded total timeout — aborting.'
            )
            if not self._replan_from_current(px, py):
                self._enter_waiting_for_path('Bug1 wall-follow total timeout')
            return

        gx, gy = self._wf_goal

        # Update leave point (closest position to goal seen so far)
        d_to_goal = math.sqrt((gx - px) ** 2 + (gy - py) ** 2)
        if d_to_goal < self._d_leave:
            self._d_leave = d_to_goal
            self._q_leave = (px, py)

        # No-progress watchdog: if the closest-to-goal distance hasn't improved
        # within the timeout, the circumnavigation is going nowhere (corner
        # oscillation / in-place spin) — abort rather than orbit forever.
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if d_to_goal < self._wf_best_d_to_goal - 0.05:
            self._wf_best_d_to_goal = d_to_goal
            self._wf_last_progress_time = now_s
        elif now_s - self._wf_last_progress_time > float(
                self.get_parameter('wall_follow_noprogress_timeout').value):
            self.get_logger().error(
                'Bug1: wall-follow made no progress — aborting.'
            )
            if not self._replan_from_current(px, py):
                self._enter_waiting_for_path('Bug1 wall-follow no progress')
            return

        # Detect when robot has moved away from hit point
        qhx, qhy = self._q_hit
        d_to_hit = math.sqrt((qhx - px) ** 2 + (qhy - py) ** 2)
        if not self._wf_left_vicinity and d_to_hit > 0.50:
            self._wf_left_vicinity = True

        # Full traversal: returned to vicinity of hit point
        if self._wf_left_vicinity and d_to_hit < 0.30:
            self.get_logger().info(
                f'Bug1: traversal complete. '
                f'Leave point: ({self._q_leave[0]:.2f}, {self._q_leave[1]:.2f}), '
                f'd_leave={self._d_leave:.2f}'
            )
            d_hit_to_goal = math.sqrt((gx - qhx) ** 2 + (gy - qhy) ** 2)
            if self._d_leave >= d_hit_to_goal - 0.05:
                self.get_logger().error('Bug1: obstacle surrounds goal — no path')
                self._enter_waiting_for_path('Bug1 obstacle surrounds goal')
                return
            self._state = _State.RETURNING_LEAVE
            self._publish_status(f'RETURNING_LEAVE: {self._current_goal_name}')
            return

        v, omega = self._compute_wall_follow_cmd(scan)
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self._cmd_vel_pub.publish(cmd)

    def _compute_wall_follow_cmd(self, scan: LaserScan) -> Tuple[float, float]:
        """Right-hand rule: keep obstacle on the right at wall_follow_dist."""
        obs_dist  = self.get_parameter('obstacle_distance').value
        wall_dist = self.get_parameter('wall_follow_dist').value
        v_max     = self.get_parameter('linear_speed').value
        w_max     = self.get_parameter('angular_max').value

        front       = self._range_at_angle(0.0,          scan, span=math.radians(30))
        front_right = self._range_at_angle(-math.pi / 4, scan, span=math.radians(20))
        right       = self._range_at_angle(-math.pi / 2, scan, span=math.radians(20))

        if front < obs_dist:
            # Blocked ahead — turn left in place
            return 0.0, w_max * 0.7

        if right > wall_dist * 2.0 and front_right > wall_dist * 1.5:
            # Wall disappeared on right (convex corner) — turn right to follow it
            return v_max * 0.5, -w_max * 0.5

        # Maintain lateral distance to right wall
        dist_error = wall_dist - right   # positive → too close → steer left
        kp_wall = 1.5
        omega = self._clamp(kp_wall * dist_error, -w_max, w_max)
        return v_max * 0.7, omega

    # ------------------------------------------------------------------
    # Bug1 — return to leave point (phase 2)
    # ------------------------------------------------------------------

    def _returning_leave_update(self, px: float, py: float, theta: float) -> None:
        """Bug1 phase 2: go to leave point then replan A*."""
        lx, ly = self._q_leave
        dx, dy = lx - px, ly - py
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # The leave point is an INTERMEDIATE waypoint, not a dock: the 3 cm
        # goal_tolerance is both semantically wrong and physically unreachable
        # here (the return speed decays below the motor deadband well before
        # 3 cm).  Use a looser band tied to the lookahead so arrival is real.
        tol = max(self.get_parameter('goal_tolerance').value,
                  0.5 * self.get_parameter('lookahead_distance').value)
        if dist < tol:
            self.get_logger().info('Bug1: arrived at leave point, replanning A*…')
            if not self._replan_from_current(px, py):
                # No path even from the closest-to-goal point on the boundary →
                # hold and keep retrying (the route may open) instead of
                # re-calling the failing replan every tick or aborting the goal.
                self.get_logger().error(
                    f'Bug1: at leave point but replan found no path for '
                    f'"{self._current_goal_name}".'
                )
                self._enter_waiting_for_path('Bug1 leave-point replan no path')
            return

        heading_error = self._wrap_angle(math.atan2(dy, dx) - theta)
        kp   = self.get_parameter('angular_kp').value
        wmax = self.get_parameter('angular_max').value
        vmax = self.get_parameter('linear_speed').value

        allow_reverse    = self.get_parameter('allow_reverse').value
        reverse_max_dist = self.get_parameter('reverse_max_dist').value
        use_reverse = (
            allow_reverse
            and abs(heading_error) > (math.pi / 2.0 + 0.1)
            and dist <= reverse_max_dist
        )
        if use_reverse:
            ctrl_error = self._wrap_angle(heading_error - math.pi)
            sign = -1.0
        else:
            ctrl_error = heading_error
            sign = +1.0

        omega = self._clamp(kp * ctrl_error, -wmax, wmax)
        speed = sign * vmax * 0.7 * max(0.0, math.cos(ctrl_error)) * min(1.0, dist / 0.3)

        # Keep the final creep above the motor deadband (mirrors FOLLOWING and
        # the heading-align floor) so the loosened leave tolerance above is
        # actually reachable instead of stalling cm short.
        approach_min = self.get_parameter('approach_speed_min').value
        if (approach_min > 0.0
                and abs(ctrl_error) < math.radians(20.0)
                and 0.0 < abs(speed) < approach_min):
            speed = sign * approach_min

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = omega
        self._cmd_vel_pub.publish(cmd)

    def _plan_with_fallback(
        self,
        planning_grid: np.ndarray,
        sx: float, sy: float,
        gx: float, gy: float,
        label: str,
    ) -> Optional[List[WorldPt]]:
        """Run A* with progressive inflation fallback.

        We first try at the configured ``inflation_radius``.  If that
        returns no path we step down — 75 %, 50 %, 25 %, 0 % — because a
        common failure mode is a goal pinned next to a wall, or rack
        legs whose inflation seals off an otherwise-clear corridor.
        Reducing inflation only changes the planner's safety margin;
        the pure-pursuit + local costmap + 360° safety bubble at runtime
        still protect against collisions.

        Returns the path (list of world points) or None if every
        inflation level failed.  Emits a single WARN with diagnostics
        when the planner gives up entirely.
        """
        infl0    = float(self.get_parameter('inflation_radius').value)
        infl_min = float(self.get_parameter('inflation_radius_min').value)
        # Coarse-to-fine ladder.  Clamp ALL rungs to `inflation_radius_min`
        # so we never produce a path the robot can't physically clear.
        # Previously the ladder ended at 0.0 which let A* return paths
        # threading between obstacles closer than the robot's own body,
        # and the robot crashed when following one of those.
        raw = [infl0, infl0 * 0.75, infl0 * 0.5, infl0 * 0.25, infl_min]
        ladder = sorted({round(max(v, infl_min), 4) for v in raw
                          if v <= infl0 + 1e-6}, reverse=True)

        for infl in ladder:
            path = astar_plan(
                planning_grid, self._origin_x, self._origin_y, self._resolution,
                sx, sy, gx, gy, inflation_radius=infl,
            )
            if path is not None:
                if infl < infl0 - 1e-6:
                    self.get_logger().warn(
                        f'A* succeeded for {label} only at reduced inflation '
                        f'{infl:.2f} m (default {infl0:.2f} m); path may be tight.'
                    )
                # Remember plan-time inflation so the validity check can
                # use a consistently tighter margin (avoids replan flapping).
                self._path_plan_infl = float(infl)
                return path

        # All fallbacks failed — emit one diagnostic line so the operator
        # can see whether the grid itself is the problem.
        try:
            occ  = int(np.count_nonzero(planning_grid == 100))
            unk  = int(np.count_nonzero(planning_grid == -1))
            free = int(planning_grid.size - occ - unk)
            self.get_logger().error(
                f'A* found no path to {label} at ANY inflation level '
                f'(tried {ladder}). Grid: {free} free / {occ} occ / {unk} unknown. '
                f'Start ({sx:.2f}, {sy:.2f}) → Goal ({gx:.2f}, {gy:.2f}). '
                f'Origin=({self._origin_x:.2f}, {self._origin_y:.2f}) '
                f'res={self._resolution}.'
            )
        except Exception:
            self.get_logger().error(f'A* found no path to {label} (diagnostics unavailable).')
        return None

    def _replan_from_current(self, px: float, py: float) -> bool:
        """Run A* (with live scan overlay) from current position to current goal.

        Returns True and transitions to FOLLOWING if a path is found.
        Returns False and leaves state unchanged if no path exists.
        """
        if self._current_goal_name not in self._waypoints:
            self._state = _State.IDLE
            self._publish_status('IDLE')
            return False

        wp = self._waypoints[self._current_goal_name]
        gx, gy = wp['x'], wp['y']

        planning_grid = self._grid_with_live_scan(px, py, self._pose_theta)
        path = self._plan_with_fallback(
            planning_grid, px, py, gx, gy,
            label=f'"{self._current_goal_name}" (replan)',
        )
        if path is None:
            return False

        self.get_logger().info(f'Replanned path ({len(path)} pts), resuming FOLLOWING')
        self._path = path
        self._last_replan_time = self.get_clock().now().nanoseconds * 1e-9
        self._reset_pd()
        self._state = _State.FOLLOWING
        self._publish_status(f'FOLLOWING: {self._current_goal_name}')
        return True

    # ------------------------------------------------------------------
    # Local costmap helpers
    # ------------------------------------------------------------------

    def _tick_costmap(self) -> None:
        """Recenter on robot pose and apply temporal decay each control tick."""
        if self._pose_x is None or self._costmap is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_costmap_tick_time is None:
            dt = 1.0 / self.get_parameter('control_rate').value
        else:
            dt = max(0.0, now - self._last_costmap_tick_time)
        self._last_costmap_tick_time = now
        self._costmap.recenter(self._pose_x, self._pose_y)
        self._costmap.decay(dt)

    def _grid_with_live_scan(self, rx: float, ry: float,
                              rtheta: float) -> np.ndarray:
        """Return a copy of the static map with the rolling costmap overlaid.

        Unlike the old single-frame overlay this uses the decay-filtered
        local costmap, so transient blips don't poison the global plan and
        persistent obstacles (rack legs, unexpected boxes) do.
        """
        if self._grid is None:
            return self._grid
        if self._costmap is None:
            return self._grid.copy()

        augmented = self._grid.copy()
        h, w = augmented.shape
        res = self._resolution
        ox, oy = self._origin_x, self._origin_y

        # Project occupied costmap cells into the global grid.
        occ = self._costmap.occupied_mask()
        if not np.any(occ):
            return augmented

        cm_side = self._costmap.side_cells
        cm_ox, cm_oy = self._costmap.origin
        cm_res = self._costmap.resolution

        # Indices of occupied cells in the costmap
        ys, xs = np.where(occ)
        # World coords of those cells
        wxs = cm_ox + (xs + 0.5) * cm_res
        wys = cm_oy + (ys + 0.5) * cm_res
        # Map to global grid indices
        gx = ((wxs - ox) / res).astype(np.int32)
        gy = ((wys - oy) / res).astype(np.int32)
        inside = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        if np.any(inside):
            augmented[gy[inside], gx[inside]] = 100

        return augmented

    # ------------------------------------------------------------------
    # Safety bubble (360°)
    # ------------------------------------------------------------------

    def _effective_safety_bubble(self) -> float:
        """Bubble radius scaled to the plan-time inflation.

        The planner placed the current path at ``_path_plan_infl`` from
        any obstacle.  Setting the bubble much wider than that would
        always fire on the very walls the planner just routed around —
        which is exactly the "Bug1 blocked" deadlock we hit at tight
        goals.  Clamp the bubble to [min, configured_max] and shrink it
        toward the plan inflation so it only fires on genuinely NEW
        obstacles closer than the planner's safety margin.
        """
        max_b = float(self.get_parameter('safety_bubble_radius').value)
        min_b = float(self.get_parameter('safety_bubble_min').value)
        return max(min_b, min(max_b, float(self._path_plan_infl)))

    def _safety_bubble_violated(self) -> bool:
        """True if at least `bubble_min_ray_count` rays in 360° fall inside
        the effective bubble.  Requiring multiple rays filters out lone
        spurious returns (glass reflections, dust, glancing hits) that
        would otherwise stop the robot in clearly open space.
        """
        scan = self._latest_scan
        if scan is None:
            return False
        radius = self._effective_safety_bubble()
        min_count = int(self.get_parameter('bubble_min_ray_count').value)
        count = 0
        for r in scan.ranges:
            if not math.isfinite(r):
                continue
            if scan.range_min < r < radius:
                count += 1
                if count >= min_count:
                    return True
        return False

    # ------------------------------------------------------------------
    # Continuous path validity check
    # ------------------------------------------------------------------

    def _path_ahead_blocked(self, px: float, py: float) -> bool:
        """Return True if the planned path within `path_validity_lookahead`
        metres ahead crosses any occupied cell in the local costmap.
        """
        if not self._path or self._costmap is None:
            return False

        lookahead = self.get_parameter('path_validity_lookahead').value
        if lookahead <= 0.0:
            return False

        # Inflation in costmap cells.  Must be STRICTLY TIGHTER than the
        # one used to plan the path: A* places path cells exactly at the
        # planner inflation boundary, so checking at the same inflation
        # would always trigger a "blocked" — even on a perfectly clean
        # plan — and cause replan flapping (the symptom we're guarding
        # against).  Scale by `path_validity_inflation_factor` (default
        # 0.5) of the inflation that produced this path.
        infl_factor = float(self.get_parameter('path_validity_inflation_factor').value)
        infl_m = max(0.0, self._path_plan_infl * infl_factor)
        infl_cells = max(0, int(round(infl_m / self._costmap.resolution)))

        # Find the closest point on the path to the robot, then walk forward
        # along the path until we've covered `lookahead` metres.
        # Use the same projection logic as _find_lookahead but accumulate
        # segments and bresenham-test each one.
        best_seg = 0
        best_t = 0.0
        best_d2 = math.inf
        for i in range(len(self._path) - 1):
            ax, ay = self._path[i]
            bx, by = self._path[i + 1]
            dx, dy = bx - ax, by - ay
            seg_len2 = dx * dx + dy * dy
            if seg_len2 < 1e-9:
                continue
            t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
            t = max(0.0, min(1.0, t))
            cx, cy = ax + t * dx, ay + t * dy
            d2 = (cx - px) ** 2 + (cy - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_seg = i
                best_t = t

        ax, ay = self._path[best_seg]
        bx, by = self._path[best_seg + 1] if best_seg + 1 < len(self._path) else self._path[best_seg]
        cur_x = ax + best_t * (bx - ax)
        cur_y = ay + best_t * (by - ay)

        remaining = lookahead
        i = best_seg
        while remaining > 0.0 and i < len(self._path) - 1:
            nx, ny = self._path[i + 1]
            seg_len = math.hypot(nx - cur_x, ny - cur_y)
            if seg_len < 1e-6:
                i += 1
                continue

            if seg_len <= remaining:
                end_x, end_y = nx, ny
                step = seg_len
            else:
                frac = remaining / seg_len
                end_x = cur_x + frac * (nx - cur_x)
                end_y = cur_y + frac * (ny - cur_y)
                step = remaining

            if self._costmap.segment_blocked(
                cur_x, cur_y, end_x, end_y, inflation_cells=infl_cells
            ):
                return True

            cur_x, cur_y = end_x, end_y
            remaining -= step
            i += 1

        return False

    # ------------------------------------------------------------------
    # Pure pursuit helpers
    # ------------------------------------------------------------------

    def _find_lookahead(self, px: float, py: float, dist: float) -> WorldPt:
        """Project robot onto the nearest path segment, advance dist metres forward."""
        if not self._path:
            return (px, py)
        if len(self._path) == 1:
            return self._path[0]

        best_seg = 0
        best_t   = 0.0
        best_d2  = math.inf

        for i in range(len(self._path) - 1):
            ax, ay = self._path[i]
            bx, by = self._path[i + 1]
            dx, dy = bx - ax, by - ay
            seg_len2 = dx * dx + dy * dy
            if seg_len2 < 1e-9:
                continue
            t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
            t = max(0.0, min(1.0, t))
            cx, cy = ax + t * dx, ay + t * dy
            d2 = (cx - px) ** 2 + (cy - py) ** 2
            if d2 < best_d2:
                best_d2  = d2
                best_seg = i
                best_t   = t

        ax, ay = self._path[best_seg]
        bx, by = self._path[best_seg + 1]
        seg_len   = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        remaining = (1.0 - best_t) * seg_len

        if remaining >= dist:
            t_new = best_t + dist / seg_len if seg_len > 1e-9 else 1.0
            return (ax + t_new * (bx - ax), ay + t_new * (by - ay))

        cumulative = remaining
        for i in range(best_seg + 1, len(self._path) - 1):
            ax2, ay2 = self._path[i]
            bx2, by2 = self._path[i + 1]
            seg2 = math.sqrt((bx2 - ax2) ** 2 + (by2 - ay2) ** 2)
            if cumulative + seg2 >= dist:
                need = dist - cumulative
                frac = need / seg2 if seg2 > 1e-9 else 1.0
                return (ax2 + frac * (bx2 - ax2), ay2 + frac * (by2 - ay2))
            cumulative += seg2

        return self._path[-1]

    # ------------------------------------------------------------------
    # Obstacle detection helpers
    # ------------------------------------------------------------------

    def _front_obstacle_detected(self) -> bool:
        """True if any scan ray in the front arc is closer than obstacle_distance."""
        scan = self._latest_scan
        if scan is None:
            return False
        threshold  = self.get_parameter('obstacle_distance').value
        half_angle = math.radians(self.get_parameter('obstacle_angle_deg').value)
        angle = scan.angle_min
        for r in scan.ranges:
            # Convert the ray bearing from the scan frame to base_link (adds
            # the LiDAR mount yaw — π on the real rear-mounted A1) so "front"
            # is genuinely the robot's +X, not the laser's.
            if abs(self._wrap_angle(angle + self._lidar_yaw)) <= half_angle:
                if scan.range_min < r < threshold:
                    return True
            angle += scan.angle_increment
        return False

    def _range_at_angle(self, target_angle: float, scan: LaserScan,
                        span: float = 0.15) -> float:
        """Average of valid ranges within span radians of target_angle."""
        angle = scan.angle_min
        vals: List[float] = []
        for r in scan.ranges:
            # target_angle is base_link-relative; bring the ray bearing into
            # base_link with the LiDAR mount yaw before comparing.
            if abs(self._wrap_angle(angle + self._lidar_yaw - target_angle)) <= span:
                if scan.range_min < r < scan.range_max:
                    vals.append(r)
            angle += scan.angle_increment
        return sum(vals) / len(vals) if vals else scan.range_max

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _on_arrived(self) -> None:
        """Position goal reached — start heading alignment if waypoint has theta."""
        self._publish_stop()
        wp = self._waypoints.get(self._current_goal_name)
        if wp and 'theta' in wp:
            self._state = _State.ALIGNING
            self._publish_status(f'ALIGNING: {self._current_goal_name}')
            self.get_logger().info(
                f'Position reached for "{self._current_goal_name}", '
                f'aligning to θ={math.degrees(wp["theta"]):.1f}°'
            )
        else:
            self._declare_arrived()

    def _aligning_update(self, theta: float) -> None:
        """Rotate in place until heading matches the waypoint's theta."""
        wp = self._waypoints.get(self._current_goal_name)
        if wp is None:
            self._declare_arrived()
            return

        target_theta  = wp['theta']
        heading_error = self._wrap_angle(target_theta - theta)
        tol = self.get_parameter('heading_tolerance').value

        if abs(heading_error) < tol:
            self._declare_arrived()
            return

        kp   = self.get_parameter('angular_kp').value
        wmax = self.get_parameter('angular_max').value
        omega = self._clamp(kp * heading_error, -wmax, wmax)

        # Ensure minimum angular speed so the robot doesn't stall just outside
        # the (now tighter) tolerance in the rotational deadband.
        min_omega = self.get_parameter('align_speed_min').value
        if min_omega > 0.0 and abs(omega) < min_omega:
            omega = math.copysign(min_omega, omega)

        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = omega
        self._cmd_vel_pub.publish(cmd)

    def _declare_arrived(self) -> None:
        """Transition to ARRIVED and schedule return to IDLE."""
        self._publish_stop()
        name = self._current_goal_name
        self._state = _State.ARRIVED
        self._publish_status(f'ARRIVED: {name}')
        self.get_logger().info(f'Arrived at "{name}" with correct heading.')
        self._arrived_timer = self.create_timer(1.0, self._arrived_timeout)

    def _cancel_arrived_timer(self) -> None:
        """Cancel a pending post-arrival → IDLE one-shot timer, if any.

        Called whenever a NEW goal is accepted or navigation is cancelled, so a
        stale arrived-timeout from the PREVIOUS goal can't fire while we're
        FOLLOWING the next one and force the node back to IDLE mid-route.
        """
        if self._arrived_timer is not None:
            self._arrived_timer.cancel()
            self._arrived_timer = None

    def _arrived_timeout(self) -> None:
        self._cancel_arrived_timer()
        self._path = []
        self._state = _State.IDLE
        self._publish_status('IDLE')

    def _cancel_navigation(self) -> None:
        self._cancel_arrived_timer()
        self._publish_stop()
        self._path = []
        self._state = _State.IDLE
        self._current_goal_name = ''
        self._path_blocked_ticks = 0
        self._reset_pd()
        self._publish_status('IDLE')
        self.get_logger().info('Navigation cancelled.')

    def _reset_pd(self) -> None:
        """Clear the pure-pursuit derivative state.

        Called wherever a fresh path/segment context begins — a new goal,
        a replan, or any re-entry into FOLLOWING.  The lookahead point (and
        therefore heading_error) can step discontinuously when self._path is
        replaced wholesale; differentiating across that step would spike
        omega.  Setting _pd_valid=False makes the next control tick rebuild
        the previous sample before contributing any derivative.
        """
        self._prev_heading_error = 0.0
        self._d_filt = 0.0
        self._pd_valid = False
        # Re-arm the FOLLOWING no-progress watchdog for this fresh path context.
        self._follow_best_dist = math.inf
        self._follow_progress_t = self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------
    # Publishing helpers
    # ------------------------------------------------------------------

    def _publish_stop(self) -> None:
        self._cmd_vel_pub.publish(Twist())

    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def _publish_plan(self) -> None:
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for wx, wy in self._path:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self._plan_pub.publish(path_msg)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _publish_map_walls(self) -> None:
        if self._grid is None:
            return
        grid = self._grid
        h, w = grid.shape
        res  = self._resolution
        ox, oy = self._origin_x, self._origin_y
        buf = bytearray()
        for row in range(h):
            for col in range(w):
                if grid[row, col] == 100:
                    x = ox + (col + 0.5) * res
                    y = oy + (row + 0.5) * res
                    buf += struct.pack('fff', x, y, 0.0)
        msg = PointCloud2()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.height    = 1
        msg.width     = len(buf) // 12
        msg.fields    = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 12
        msg.row_step     = len(buf)
        msg.data         = bytes(buf)
        msg.is_dense     = True
        self._walls_pub.publish(msg)

    def _publish_waypoint_markers(self) -> None:
        array = MarkerArray()
        now   = self.get_clock().now().to_msg()
        clear = Marker()
        clear.action          = Marker.DELETEALL
        clear.header.frame_id = 'map'
        clear.header.stamp    = now
        array.markers.append(clear)
        for idx, (label, wp) in enumerate(self._waypoints.items()):
            x, y, theta = wp['x'], wp['y'], wp['theta']
            half = theta / 2.0
            qz, qw = math.sin(half), math.cos(half)

            arrow            = Marker()
            arrow.header.frame_id = 'map'
            arrow.header.stamp    = now
            arrow.ns, arrow.id    = 'wp_arrows', idx * 2
            arrow.type            = Marker.ARROW
            arrow.action          = Marker.ADD
            arrow.pose.position.x = x
            arrow.pose.position.y = y
            arrow.pose.position.z = 0.05
            arrow.pose.orientation.z = qz
            arrow.pose.orientation.w = qw
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.5, 0.08, 0.08
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 0.1, 0.9, 0.3, 1.0
            array.markers.append(arrow)

            text             = Marker()
            text.header.frame_id = 'map'
            text.header.stamp    = now
            text.ns, text.id     = 'wp_labels', idx * 2 + 1
            text.type            = Marker.TEXT_VIEW_FACING
            text.action          = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = 0.4
            text.pose.orientation.w = 1.0
            text.scale.z            = 0.25
            text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 1.0
            text.text = label
            array.markers.append(text)
        self._markers_pub.publish(array)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
