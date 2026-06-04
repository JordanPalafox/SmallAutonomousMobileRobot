"""
state_machine_node.py
---------------------
ROS2 node hosting the YASMIN state machine for the AMR warehouse robot.

I/O contract
~~~~~~~~~~~~
Subscribes
    /mission             (String, JSON)         — mission requests
    /nav_status          (String)               — from nav_node
    /alignment_state     (String)               — from qr_quad_alignment
    /alignment_error     (Point)                — kept for legacy consumers
    /qr_detected         (String)               — from qr_quad_alignment
    /lifter_status       (UInt8)                — from lifting_node
    /voice_command       (String)               — from voice_control
    /sm/control          (String, JSON)         — debugger commands

Publishes
    /goal_waypoint       (String)               — to nav_node
    /lifter_level        (UInt8)                — to lifting_node
    /alignment_start     (Bool)                 — to qr_quad_alignment
    /robot_state         (String)               — current state name
    /sm/blackboard       (String, JSON)         — periodic snapshot
    /sm/transition       (String, JSON)         — one message per transition

Debugger (/sm/control payloads)
    {"action": "pause"}
    {"action": "resume"}
    {"action": "step"}
    {"action": "set_step_mode", "value": true|false}
    {"action": "force_outcome", "value": "<outcome>"}
    {"action": "abort"}
    {"action": "clear_abort"}
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String, UInt8

from yasmin import Blackboard, StateMachine

from mission_control.debug_wrapper import DebugContext
from mission_control.bb_helpers import bb_get
from mission_control.mission_parser import load_zones, derive_zones
from mission_control.states.idle import Idle
from mission_control.states.mission_done import MissionDone
from mission_control.states.mission_failed import MissionFailed
from mission_control.states.nav_to_truck import NavToTruck
from mission_control.states.pick import Pick
from mission_control.states.place import Place
from mission_control.states.search import Search

logger = logging.getLogger(__name__)


def _reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        depth=depth,
    )


class StateMachineNode(Node):
    SNAPSHOT_HZ = 2.0

    def __init__(self) -> None:
        super().__init__("state_machine_node")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("zones_file", "")
        self.declare_parameter("waypoints_file", "~/ros2_maps/waypoints.yaml")
        self.declare_parameter("scan_qr_timeout", 4.0)
        self.declare_parameter("alignment_timeout", 30.0)
        self.declare_parameter("lifter_timeout", 8.0)
        self.declare_parameter("debug_step_mode_default", False)
        # ── PICK (timed open-loop maneuver) ────────────────────────────────
        # After QR docking: lift to entry_level, drive forward forward_time s to
        # slide the forks under the pallet, change to lift_level to take its
        # weight, then reverse reverse_time s to pull it clear.
        self.declare_parameter("pick_approach_speed", 0.10)    # m/s forward & reverse
        self.declare_parameter("pick_forward_time", 5.0)       # s driving into the pallet
        self.declare_parameter("pick_reverse_time", 10.0)      # s backing out
        self.declare_parameter("pick_entry_level", 1)          # fork height to enter
        self.declare_parameter("pick_lift_level", 0)           # height to lift the pallet
        self.declare_parameter("pick_transport_level", 1)      # lifter height after backing out
        # Stall detection for the forward-into-pallet move: stop the instant the
        # wheels can't turn (robot blocked by the pallet) so the motor driver
        # isn't left drawing stall current. Stall = wheel speed (/wl,/wr rad/s)
        # below pick_stall_speed for pick_stall_ticks ticks, after a spin-up grace.
        self.declare_parameter("pick_stall_grace", 1.0)        # s spin-up before checking
        self.declare_parameter("pick_stall_speed", 0.4)        # rad/s — below = stalled
        self.declare_parameter("pick_stall_ticks", 3)          # consecutive ticks to confirm
        # VISION stop for the forward creep: stop when the Electric-80 logo
        # detector (perception/logo_stop_debug → /approach_stop/should_stop)
        # says the load is at the target distance, BEFORE contact. This is the
        # brownout fix — stall above is only the fallback. pick_vision_fresh_s
        # = max age of the should_stop signal we trust (ignore a frozen value
        # from a dead detector).
        self.declare_parameter("pick_vision_stop", True)
        self.declare_parameter("pick_vision_fresh_s", 1.0)
        # Front-distance TELEMETRY only (shown in the dashboard; does NOT gate
        # the timed maneuver). Front arc + LiDAR mount yaw select the front rays.
        self.declare_parameter("pick_front_arc_deg", 12.0)
        self.declare_parameter("laser_yaw_offset_deg", 180.0)  # rear-mounted LiDAR
        self.declare_parameter("pick_front_min_range", 0.12)

        zones_path = str(self.get_parameter("zones_file").value)
        if not zones_path:
            zones_path = os.path.join(
                get_package_share_directory("mission_control"), "config", "zones.yaml"
            )
        self._zones = load_zones(zones_path)         # qr_aliases (+ optional zones)
        self.get_logger().info(f"Loaded qr_aliases from {zones_path}")

        # Zone classification is derived from the actual waypoint names by prefix
        # (roller_/rack_/truck_), so the dashboard's auto-named waypoints define
        # the zones with zero manual sync. Falls back to any zones listed in
        # zones.yaml if no waypoints file is found.
        wp_path = str(self.get_parameter("waypoints_file").value)
        wp_names = self._load_waypoint_names(wp_path)
        if wp_names:
            self._zones["zones"] = derive_zones(wp_names)
            self.get_logger().info(
                f"Derived zones from {len(wp_names)} waypoints in {wp_path}: "
                f"{ {k: len(v) for k, v in self._zones['zones'].items()} }"
            )
        else:
            self.get_logger().warning(
                f"No waypoints found at {wp_path}; using zones from {zones_path} (may be empty)."
            )

        scan_qr_timeout   = float(self.get_parameter("scan_qr_timeout").value)
        alignment_timeout = float(self.get_parameter("alignment_timeout").value)
        lifter_timeout    = float(self.get_parameter("lifter_timeout").value)
        step_default      = bool(self.get_parameter("debug_step_mode_default").value)

        import math as _math
        pick_approach_speed    = float(self.get_parameter("pick_approach_speed").value)
        pick_forward_time      = float(self.get_parameter("pick_forward_time").value)
        pick_reverse_time      = float(self.get_parameter("pick_reverse_time").value)
        pick_entry_level       = int(self.get_parameter("pick_entry_level").value)
        pick_lift_level        = int(self.get_parameter("pick_lift_level").value)
        pick_transport_level   = int(self.get_parameter("pick_transport_level").value)
        pick_stall_grace       = float(self.get_parameter("pick_stall_grace").value)
        pick_stall_speed       = float(self.get_parameter("pick_stall_speed").value)
        pick_stall_ticks       = int(self.get_parameter("pick_stall_ticks").value)
        pick_vision_stop       = bool(self.get_parameter("pick_vision_stop").value)
        pick_vision_fresh_s    = float(self.get_parameter("pick_vision_fresh_s").value)
        self._pick_front_arc   = _math.radians(float(self.get_parameter("pick_front_arc_deg").value))
        self._laser_yaw_offset = _math.radians(float(self.get_parameter("laser_yaw_offset_deg").value))
        self._pick_front_min_range = float(self.get_parameter("pick_front_min_range").value)

        # ── Shared state ──────────────────────────────────────────────────
        self._blackboard = Blackboard()
        self._init_blackboard()
        self._debug = DebugContext()
        if step_default:
            self._debug.set_step_mode(True)
        self._current_state_name: str = "IDLE"

        # ── ROS I/O ───────────────────────────────────────────────────────
        qos = _reliable_qos()

        self.create_subscription(String, "/mission",          self._cb_mission,        qos)
        self.create_subscription(String, "/nav_status",       self._cb_nav_status,     qos)
        self.create_subscription(String, "/alignment_state",  self._cb_alignment_state, qos)
        self.create_subscription(Point,  "/alignment_error",  self._cb_alignment_error, qos)
        self.create_subscription(String, "/qr_detected",      self._cb_qr_detected,    qos)
        self.create_subscription(UInt8,  "/lifter_status",    self._cb_lifter_status,  qos)
        self.create_subscription(String, "/voice_command",    self._cb_voice,          qos)
        self.create_subscription(String, "/sm/control",       self._cb_sm_control,     qos)
        # /scan for the PICK forward-approach (best-effort sensor QoS).
        self.create_subscription(LaserScan, "/scan", self._cb_scan, qos_profile_sensor_data)
        # Wheel encoder speeds (rad/s) → stall detection during the PICK forward.
        self._wl = 0.0
        self._wr = 0.0
        self.create_subscription(Float32, "/wl", self._cb_wl, qos_profile_sensor_data)
        self.create_subscription(Float32, "/wr", self._cb_wr, qos_profile_sensor_data)
        # Vision stop signal for the PICK forward creep (logo_stop_debug on the
        # Jetson). Best-effort for freshest delivery over WiFi (the detector's
        # RELIABLE publisher is compatible with a BEST_EFFORT subscription).
        self.create_subscription(Bool, "/approach_stop/should_stop",
                                 self._cb_approach_stop, qos_profile_sensor_data)

        self._pub_goal       = self.create_publisher(String, "/goal_waypoint",   qos)
        self._pub_cmd        = self.create_publisher(Twist,  "/cmd_vel_in",      qos)
        self._pub_lifter     = self.create_publisher(UInt8,  "/lifter_level",    qos)
        self._pub_align      = self.create_publisher(Bool,   "/alignment_start", qos)
        self._pub_state      = self.create_publisher(String, "/robot_state",     qos)
        self._pub_blackboard = self.create_publisher(String, "/sm/blackboard",   qos)
        self._pub_transition = self.create_publisher(String, "/sm/transition",   qos)

        # ── Build SM ──────────────────────────────────────────────────────
        self._sm = self._build_state_machine(
            scan_qr_timeout=scan_qr_timeout,
            alignment_timeout=alignment_timeout,
            lifter_timeout=lifter_timeout,
            pick_approach_speed=pick_approach_speed,
            pick_forward_time=pick_forward_time,
            pick_reverse_time=pick_reverse_time,
            pick_entry_level=pick_entry_level,
            pick_lift_level=pick_lift_level,
            pick_stall_grace=pick_stall_grace,
            pick_stall_speed=pick_stall_speed,
            pick_stall_ticks=pick_stall_ticks,
            pick_vision_stop=pick_vision_stop,
            pick_vision_fresh_s=pick_vision_fresh_s,
            pick_transport_level=pick_transport_level,
        )
        self._sm.set_start_state("IDLE")

        # ── Periodic snapshot ─────────────────────────────────────────────
        self.create_timer(1.0 / self.SNAPSHOT_HZ, self._publish_snapshot)

        # ── Start SM thread ───────────────────────────────────────────────
        self._sm_thread = threading.Thread(
            target=self._run_state_machine, daemon=True, name="sm_thread",
        )
        self._sm_thread.start()
        self.get_logger().info("StateMachineNode started.")

    # ------------------------------------------------------------------ init
    def _init_blackboard(self) -> None:
        bb = self._blackboard
        bb["mission_raw"]         = None
        bb["current_mission"]     = None
        bb["candidate_queue"]     = None
        bb["current_candidate"]   = None
        bb["qr_value"]            = None
        bb["qr_detected"]         = None
        bb["qr_detected_at"]      = 0.0
        bb["resolved_dest"]       = None
        bb["nav_status_prefix"]   = "IDLE"
        bb["nav_status_full"]     = "IDLE"
        bb["alignment_state"]     = "IDLE"
        bb["alignment_error"]     = None
        bb["lifter_status"]       = None
        bb["voice_command"]       = None
        bb["mission_error_reason"] = None
        bb["front_distance"]       = None    # min front-arc LiDAR range (m), telemetry
        bb["wheel_speed"]          = None    # max |wheel| rad/s, for PICK stall detection
        # Vision stop for the PICK creep, stored as ONE tuple (should_stop, stamp)
        # so the boolean and its monotonic timestamp are always read consistently
        # across the executor and SM threads (no torn read of two separate keys).
        bb["approach_stop_signal"] = None

    # ------------------------------------------------------------------ waypoints
    @staticmethod
    def _load_waypoint_names(path: str) -> list[str]:
        """Read waypoint names from a waypoints.yaml ({name: {x,y,theta}})."""
        import yaml
        path = os.path.expanduser(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            wps = data.get("waypoints", data)
            return list(wps.keys()) if isinstance(wps, dict) else []
        except FileNotFoundError:
            return []
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------ callbacks
    def _cb_mission(self, msg: String) -> None:
        if bb_get(self._blackboard, "current_mission") is not None:
            self.get_logger().warn(
                "Ignoring mission — one is already running. Send abort first."
            )
            return
        self._blackboard["mission_raw"] = msg.data
        self.get_logger().info(f"Mission queued: {msg.data[:120]}")

    def _cb_nav_status(self, msg: String) -> None:
        raw = msg.data.strip()
        prefix = raw.split(":", 1)[0].upper()
        self._blackboard["nav_status_prefix"] = prefix
        self._blackboard["nav_status_full"] = raw

    def _cb_alignment_state(self, msg: String) -> None:
        self._blackboard["alignment_state"] = msg.data.strip().upper()

    def _cb_alignment_error(self, msg: Point) -> None:
        self._blackboard["alignment_error"] = msg

    def _cb_qr_detected(self, msg: String) -> None:
        payload = msg.data.strip()
        if not payload:
            return
        self._blackboard["qr_detected"] = payload
        self._blackboard["qr_detected_at"] = time.monotonic()

    def _cb_lifter_status(self, msg: UInt8) -> None:
        self._blackboard["lifter_status"] = int(msg.data)

    def _cb_scan(self, msg: LaserScan) -> None:
        """Min valid range in the front arc (base_link forward) for PICK approach.

        Each ray bearing is brought into base_link with the LiDAR mount yaw
        (rear-mounted → 180°), then only rays within ±front_arc of straight-ahead
        and within [range_min, range_max] are considered. None if no valid ray.
        """
        import math
        angle = msg.angle_min
        # Floor out very-near returns (robot's own structure / forks / noise) so
        # the front distance reflects the pallet, not the robot itself.
        floor = max(msg.range_min, self._pick_front_min_range)
        rmax = msg.range_max
        arc, yaw = self._pick_front_arc, self._laser_yaw_offset
        best = None
        for r in msg.ranges:
            aw = math.atan2(math.sin(angle + yaw), math.cos(angle + yaw))
            if -arc <= aw <= arc and floor < r < rmax:
                if best is None or r < best:
                    best = r
            angle += msg.angle_increment
        self._blackboard["front_distance"] = best

    def _cb_wl(self, msg: Float32) -> None:
        self._wl = float(msg.data)
        self._blackboard["wheel_speed"] = max(abs(self._wl), abs(self._wr))

    def _cb_wr(self, msg: Float32) -> None:
        self._wr = float(msg.data)
        self._blackboard["wheel_speed"] = max(abs(self._wl), abs(self._wr))

    def _cb_approach_stop(self, msg: Bool) -> None:
        """Vision stop for the PICK creep (logo at target distance).

        Written as a single (should_stop, stamp) tuple so a reader in the SM
        thread can never see a fresh bool paired with a stale timestamp.
        """
        self._blackboard["approach_stop_signal"] = (bool(msg.data), time.monotonic())

    def _cb_voice(self, msg: String) -> None:
        cmd = msg.data.strip().lower()
        self._blackboard["voice_command"] = cmd
        if cmd in ("stop", "pause"):
            self._debug.set_pause(True)
        elif cmd == "resume":
            self._debug.set_pause(False)

    def _cb_sm_control(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"/sm/control: invalid JSON: {msg.data!r}")
            return
        action = payload.get("action")
        if action == "pause":
            self._debug.set_pause(True)
        elif action == "resume":
            self._debug.set_pause(False)
        elif action == "step":
            self._debug.request_step()
        elif action == "set_step_mode":
            self._debug.set_step_mode(bool(payload.get("value", False)))
        elif action == "force_outcome":
            self._debug.set_force_outcome(payload.get("value"))
        elif action == "abort":
            self._debug.set_abort(True)
            # Make sure the robot doesn't keep moving while in abort.
            self._publish_goal("stop")
            self._publish_alignment_start(False)
        elif action == "clear_abort":
            self._debug.set_abort(False)
        else:
            self.get_logger().warn(f"/sm/control: unknown action {action!r}")
            return
        self.get_logger().info(f"/sm/control: applied {payload}")

    # ------------------------------------------------------------------ outputs
    def _publish_goal(self, waypoint_name: str) -> None:
        m = String(); m.data = waypoint_name
        self._pub_goal.publish(m)

    def _publish_cmd(self, v: float, w: float) -> None:
        """Drive command for the PICK forward-approach (→ /cmd_vel_in)."""
        m = Twist(); m.linear.x = float(v); m.angular.z = float(w)
        self._pub_cmd.publish(m)

    def _publish_lifter(self, level: int) -> None:
        m = UInt8(); m.data = max(0, min(7, int(level)))
        self._pub_lifter.publish(m)

    def _publish_alignment_start(self, start: bool) -> None:
        m = Bool(); m.data = bool(start)
        self._pub_align.publish(m)

    def _publish_state(self, state_name: str) -> None:
        self._current_state_name = state_name
        m = String(); m.data = state_name
        self._pub_state.publish(m)
        self.get_logger().info(f"→ {state_name}")

    def _publish_transition(self, state_name: str, outcome: str) -> None:
        payload = {
            "t": self.get_clock().now().nanoseconds / 1e9,
            "state": state_name,
            "outcome": outcome,
        }
        m = String(); m.data = json.dumps(payload)
        self._pub_transition.publish(m)

    def _publish_snapshot(self) -> None:
        m = String(); m.data = json.dumps(self._snapshot_blackboard())
        self._pub_blackboard.publish(m)

    def _snapshot_blackboard(self) -> dict:
        bb = self._blackboard
        mission = bb.get("current_mission")
        queue = bb.get("candidate_queue")
        ae = bb.get("alignment_error")
        sig = bb.get("approach_stop_signal")
        return {
            "state":              self._current_state_name,
            "mission_id":         (mission or {}).get("id"),
            "mission_type":       (mission or {}).get("type"),
            "candidate_queue":    list(queue) if isinstance(queue, deque) else None,
            "current_candidate":  bb.get("current_candidate"),
            "qr_detected":        bb.get("qr_detected"),
            "qr_value":           bb.get("qr_value"),
            "resolved_dest":      bb.get("resolved_dest"),
            "nav_status":         bb.get("nav_status_full"),
            "alignment_state":    bb.get("alignment_state"),
            "alignment_error":    None if ae is None else {"x": ae.x, "y": ae.y, "z": ae.z},
            "lifter_status":      bb.get("lifter_status"),
            "voice_command":      bb.get("voice_command"),
            "front_distance":     bb.get("front_distance"),   # min front-arc LiDAR range (m)
            "wheel_speed":        bb.get("wheel_speed"),      # max |wheel| rad/s (stall sense)
            "approach_should_stop": (sig[0] if sig else None),  # vision PICK stop

            "mission_error_reason": bb.get("mission_error_reason"),
            "debug":              self._debug.snapshot(),
        }

    # ------------------------------------------------------------------ build SM
    def _build_state_machine(
        self,
        scan_qr_timeout: float,
        alignment_timeout: float,
        lifter_timeout: float,
        pick_approach_speed: float,
        pick_forward_time: float,
        pick_reverse_time: float,
        pick_entry_level: int,
        pick_lift_level: int,
        pick_stall_grace: float,
        pick_stall_speed: float,
        pick_stall_ticks: int,
        pick_vision_stop: bool,
        pick_vision_fresh_s: float,
        pick_transport_level: int,
    ) -> StateMachine:
        sm = StateMachine(outcomes=["finish"])
        kw = dict(
            on_enter=self._publish_state,
            on_transition=self._publish_transition,
        )

        sm.add_state(
            "IDLE",
            Idle(self._debug, self._zones, **kw),
            transitions={"mission_received": "SEARCH"},
        )
        # STEP 1 — buscar el pallet: navigate candidates in order, read the QR.
        sm.add_state(
            "SEARCH",
            Search(self._debug, self._publish_goal, scan_qr_timeout, **kw),
            transitions={
                "found":     "PICK",
                "done":      "MISSION_DONE",   # SEARCH_ROLLERS / SEARCH_RACKS: stop after finding
                "not_found": "MISSION_FAILED",
                "stop":      "MISSION_FAILED",
            },
        )
        # STEP 2 — recoge: align to the pallet, then lift it.
        sm.add_state(
            "PICK",
            Pick(self._debug, self._publish_alignment_start, self._publish_lifter,
                 self._publish_cmd, alignment_timeout, lifter_timeout,
                 pick_approach_speed, pick_forward_time, pick_reverse_time,
                 pick_entry_level, pick_lift_level,
                 pick_stall_grace, pick_stall_speed, pick_stall_ticks,
                 pick_vision_stop, pick_vision_fresh_s, pick_transport_level, **kw),
            transitions={
                "picked": "NAV_TO_TRUCK",
                "done":   "MISSION_DONE",     # PICK_ONLY test ends here
                "failed": "MISSION_FAILED",
                "stop":   "MISSION_FAILED",
            },
        )
        # STEP 3 — navega al camión según el QR (resolve inline + drive).
        sm.add_state(
            "NAV_TO_TRUCK",
            NavToTruck(self._debug, self._publish_goal, self._zones, **kw),
            transitions={
                "arrived": "PLACE",
                "failed":  "MISSION_FAILED",
                "stop":    "MISSION_FAILED",
            },
        )
        # STEP 4 — deja: lower the lifter to release the pallet.
        sm.add_state(
            "PLACE",
            Place(self._debug, self._publish_lifter, lifter_timeout, **kw),
            transitions={
                "placed": "MISSION_DONE",
                "failed": "MISSION_FAILED",
                "stop":   "MISSION_FAILED",
            },
        )
        sm.add_state(
            "MISSION_DONE",
            MissionDone(self._debug, **kw),
            transitions={"ok": "IDLE"},
        )
        sm.add_state(
            "MISSION_FAILED",
            MissionFailed(self._debug, self._publish_goal, self._publish_alignment_start, **kw),
            transitions={"ok": "IDLE"},
        )

        return sm

    # ------------------------------------------------------------------ thread
    def _run_state_machine(self) -> None:
        import traceback
        while rclpy.ok():
            try:
                # yasmin StateMachine is invoked via __call__, not .execute.
                self._sm(self._blackboard)
            except Exception:  # noqa: BLE001
                self.get_logger().error(
                    "State machine raised — restarting in 1s.\n" + traceback.format_exc()
                )
                time.sleep(1.0)


# ----------------------------------------------------------------------------
def main() -> None:
    rclpy.init()
    node = StateMachineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
