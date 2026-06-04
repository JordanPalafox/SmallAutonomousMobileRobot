"""
RosBridge — thread-safe bridge between ROS2 subscribers and the Flask web app.

Subscriptions run in the ROS2 executor thread.
Flask reads data via thread-safe locks.
"""

import collections
import io
import json
import os
import struct
import threading
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
)

# Number of recent SM transitions kept for the dashboard log.
SM_TRANSITION_BUFFER = 20


@dataclass
class RobotState:
    pose: dict = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "theta": 0.0})
    state: str = "UNKNOWN"             # mission_control SM state (was /robot_state)
    nav_status: str = "IDLE"           # nav_node status (was state, kept separate now)
    mission: Optional[dict] = None
    velocity: dict = field(default_factory=lambda: {"linear": 0.0, "angular": 0.0})
    lifter_level: int = 0
    scan: Optional[dict] = None
    nav_plan: Optional[list] = None
    system_mode: str = "NAVIGATION"


class RosBridge:
    """
    Thread-safe bridge between ROS2 and Flask.

    All ROS2 callbacks write to shared state under a lock.
    Flask threads read that state via get_state() / get_latest_frame().
    """

    def __init__(self, node) -> None:
        self._node = node
        self._lock = threading.Lock()
        self._state = RobotState()
        self._latest_frame: Optional[bytes] = None  # JPEG bytes
        self._latest_map: Optional[bytes] = None    # PNG bytes
        self._map_info: dict = {}                   # origin, resolution, size, source
        self._waypoints: dict = {}
        # mission_control debug surface
        self._sm_snapshot: Optional[dict] = None
        self._sm_transitions: collections.deque = collections.deque(
            maxlen=SM_TRANSITION_BUFFER,
        )

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Lazy imports — ROS2 message types only available at runtime
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import OccupancyGrid, Path
        from sensor_msgs.msg import Image, LaserScan
        from std_msgs.msg import String, UInt8

        self._node.create_subscription(
            PoseStamped, "/slam_pose", self._cb_pose, best_effort_qos
        )
        self._node.create_subscription(
            String, "/nav_status", self._cb_nav_status, reliable_qos
        )
        self._node.create_subscription(
            String, "/robot_state", self._cb_robot_state, reliable_qos
        )
        self._node.create_subscription(
            String, "/sm/blackboard", self._cb_sm_blackboard, reliable_qos
        )
        self._node.create_subscription(
            String, "/sm/transition", self._cb_sm_transition, reliable_qos
        )
        self._node.create_subscription(
            String, "/mission", self._cb_mission, reliable_qos
        )
        self._node.create_subscription(
            Twist, "/cmd_vel", self._cb_vel, best_effort_qos
        )
        self._node.create_subscription(
            UInt8, "/lifter_status", self._cb_lifter, best_effort_qos
        )
        # Camera topic is configurable. Default = the QR detector's annotated
        # stream (/qr_quad_alignment/debug_image), so the dashboard shows live
        # QR detection + pose overlay. For the raw camera instead, set
        # camera_topic:=/video_source/raw (real robot) or /mast_camera/image_raw (sim).
        self._node.declare_parameter("camera_topic", "/qr_quad_alignment/debug_image")
        camera_topic = str(self._node.get_parameter("camera_topic").value)
        self._node.get_logger().info(f"Camera feed subscribed on {camera_topic}")
        self._node.create_subscription(
            Image, camera_topic, self._cb_image, best_effort_qos
        )
        self._node.create_subscription(
            OccupancyGrid, "/map", self._cb_map, best_effort_qos
        )
        self._node.create_subscription(
            LaserScan, "/scan", self._cb_scan, best_effort_qos
        )
        self._node.create_subscription(
            Path, "/plan", self._cb_plan, reliable_qos
        )

        self._mission_pub = self._node.create_publisher(String, "/mission", reliable_qos)
        self._goal_pub = self._node.create_publisher(String, "/goal_waypoint", reliable_qos)
        self._lifter_pub = self._node.create_publisher(UInt8, "/lifter_level", reliable_qos)
        self._cmd_vel_pub = self._node.create_publisher(Twist, "/cmd_vel_in", reliable_qos)
        self._voice_pub = self._node.create_publisher(String, "/voice_command", reliable_qos)
        self._sm_control_pub = self._node.create_publisher(String, "/sm/control", reliable_qos)

        # /system_mode is latched (transient_local) so late subscribers see the
        # current mode immediately.
        mode_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._mode_pub = self._node.create_publisher(String, "/system_mode", mode_qos)
        # Echo back so we keep state.system_mode in sync (in case another node
        # publishes too).
        self._node.create_subscription(
            String, "/system_mode", self._cb_system_mode, mode_qos
        )

        # Service clients for slam_node (~/save_map, ~/reset_map) and nav_node
        # (~/reload_waypoints).  They're created here but called lazily so the
        # dashboard works even if those services aren't up yet.
        from std_srvs.srv import Trigger
        self._save_map_cli   = self._node.create_client(Trigger, '/map_saver/save_map')
        self._reset_map_cli  = self._node.create_client(Trigger, '/slam_node/reset_map')
        self._reload_wps_cli = self._node.create_client(Trigger, '/nav_node/reload_waypoints')

        # Load waypoints and static map from file at startup
        self._load_waypoints()
        self._load_static_map()

        # Publish the initial system mode (read by app.py via set_system_mode).
        # Default is NAVIGATION — the launch file overrides this through the
        # `start_mode` parameter on the dashboard node.
        initial_mode = "NAVIGATION"
        try:
            self._node.declare_parameter('start_mode', 'navigation')
            sm = str(self._node.get_parameter('start_mode').value).lower()
            if sm == 'mapping':
                initial_mode = 'MAPPING'
        except Exception:
            pass
        self.publish_system_mode(initial_mode)

    # ------------------------------------------------------------------
    # Waypoints loading
    # ------------------------------------------------------------------

    def _load_waypoints(self) -> None:
        """Load waypoints.yaml from ~/ros2_maps/ at startup."""
        import yaml

        waypoints_path = os.path.expanduser("~/ros2_maps/waypoints.yaml")
        try:
            with open(waypoints_path, "r") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                self._waypoints = data.get("waypoints", data)
                self._node.get_logger().info(
                    f"Loaded {len(self._waypoints)} waypoints from {waypoints_path}"
                )
            else:
                self._node.get_logger().warning(
                    f"Unexpected waypoints format in {waypoints_path}"
                )
        except FileNotFoundError:
            self._node.get_logger().warning(
                f"Waypoints file not found: {waypoints_path} — using empty waypoints"
            )
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warning(f"Could not load waypoints: {exc}")

    # ------------------------------------------------------------------
    # Public API (called from Flask threads)
    # ------------------------------------------------------------------

    def get_state(self) -> RobotState:
        with self._lock:
            import copy
            return copy.deepcopy(self._state)

    def get_latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    def get_latest_map(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_map

    def get_map_info(self) -> dict:
        with self._lock:
            return dict(self._map_info)

    def publish_mission(self, mission_json: str) -> None:
        from std_msgs.msg import String
        msg = String()
        msg.data = mission_json
        self._mission_pub.publish(msg)

    def publish_goal(self, name: str) -> None:
        """Publish a goal waypoint name to /goal_waypoint."""
        from std_msgs.msg import String
        msg = String()
        msg.data = name
        self._goal_pub.publish(msg)

    def publish_voice_command(self, word: str) -> None:
        """Publish a recognised word to /voice_command."""
        from std_msgs.msg import String
        msg = String()
        msg.data = word
        self._voice_pub.publish(msg)

    def publish_lifter_level(self, level: int) -> None:
        """Publish a target lifter level to /lifter_level (clamped 0-7; the
        lifting node further clamps to the active HAL's range)."""
        from std_msgs.msg import UInt8
        msg = UInt8()
        msg.data = max(0, min(7, int(level)))
        self._lifter_pub.publish(msg)

    def get_sm_snapshot(self) -> Optional[dict]:
        with self._lock:
            return dict(self._sm_snapshot) if self._sm_snapshot is not None else None

    def get_sm_transitions(self) -> list:
        with self._lock:
            return list(self._sm_transitions)

    def publish_sm_control(self, payload: dict) -> None:
        """Send a debugger command to mission_control on /sm/control.

        Expected payloads (see state_machine_node docstring):
            {"action": "pause"} / "resume" / "step" / "abort" / "clear_abort"
            {"action": "set_step_mode", "value": true|false}
            {"action": "force_outcome", "value": "<outcome>"}
        """
        from std_msgs.msg import String
        msg = String()
        msg.data = json.dumps(payload)
        self._sm_control_pub.publish(msg)

    def publish_cmd_vel(self, linear: float, angular: float) -> None:
        """Publish a Twist command to /cmd_vel."""
        from geometry_msgs.msg import Twist
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._cmd_vel_pub.publish(msg)

    def get_waypoints(self) -> dict:
        """Return the waypoints dict {name: {x, y, theta}}."""
        return dict(self._waypoints)

    def add_typed_waypoint(self, wtype: str, x: float, y: float, theta: float) -> str:
        """Add a waypoint with an auto-assigned name based on its type.

        The name follows the ``<type>_<n>`` convention (roller_1, rack_2,
        truck_3) that the rest of the stack already keys off (map colours,
        zone classification). The next index is computed under the lock so two
        near-simultaneous saves can't pick the same name. Only geometry
        (x, y, theta) is persisted — the type stays encoded in the name prefix,
        keeping waypoints.yaml compatible with nav_node's loader.

        Returns the assigned waypoint name.
        """
        import re
        with self._lock:
            pattern = re.compile(rf"^{re.escape(wtype)}_(\d+)$")
            max_n = 0
            for key in self._waypoints:
                m = pattern.match(key)
                if m:
                    max_n = max(max_n, int(m.group(1)))
            name = f"{wtype}_{max_n + 1}"
        self.add_waypoint(name, x, y, theta)
        return name

    def add_waypoint(self, name: str, x: float, y: float, theta: float) -> None:
        """Add or update a waypoint and persist to the YAML file."""
        import yaml
        with self._lock:
            self._waypoints[name] = {"x": x, "y": y, "theta": theta}
            snapshot = dict(self._waypoints)
        waypoints_path = os.path.expanduser("~/ros2_maps/waypoints.yaml")
        try:
            os.makedirs(os.path.dirname(waypoints_path), exist_ok=True)
            with open(waypoints_path, "w") as f:
                yaml.dump({"waypoints": snapshot}, f, default_flow_style=False)
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warning(f"Could not save waypoints: {exc}")
        # Tell nav_node to reload waypoints so the new one is immediately
        # selectable as a goal.
        self.call_reload_waypoints()

    def delete_waypoint(self, name: str) -> bool:
        """Remove a waypoint and persist to disk.  Returns True if removed."""
        import yaml
        with self._lock:
            if name not in self._waypoints:
                return False
            del self._waypoints[name]
            snapshot = dict(self._waypoints)
        waypoints_path = os.path.expanduser("~/ros2_maps/waypoints.yaml")
        try:
            with open(waypoints_path, "w") as f:
                yaml.dump({"waypoints": snapshot}, f, default_flow_style=False)
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warning(f"Could not save waypoints: {exc}")
        self.call_reload_waypoints()
        return True

    # ------------------------------------------------------------------
    # System mode + slam services
    # ------------------------------------------------------------------

    def publish_system_mode(self, mode: str) -> None:
        """Publish /system_mode with a latched QoS so late subscribers see it."""
        from std_msgs.msg import String
        mode = mode.strip().upper()
        if mode not in ('MAPPING', 'NAVIGATION'):
            self._node.get_logger().warning(f"Refusing to publish invalid mode {mode}")
            return
        msg = String()
        msg.data = mode
        self._mode_pub.publish(msg)
        with self._lock:
            self._state.system_mode = mode
        self._node.get_logger().info(f"/system_mode published: {mode}")

    def get_system_mode(self) -> str:
        with self._lock:
            return self._state.system_mode

    def _call_trigger(self, client, name: str, timeout: float = 2.0) -> tuple[bool, str]:
        """Call a Trigger-typed service, blocking up to `timeout` seconds."""
        from std_srvs.srv import Trigger
        if not client.wait_for_service(timeout_sec=timeout):
            return False, f"{name}: service not available"
        req = Trigger.Request()
        future = client.call_async(req)
        # Spin briefly waiting for the response.  We're already on the executor
        # thread for some callers, so use the executor's spin_until_future_complete.
        import time
        t0 = time.monotonic()
        while not future.done() and (time.monotonic() - t0) < timeout:
            time.sleep(0.05)
        if not future.done():
            return False, f"{name}: timed out after {timeout:.1f}s"
        try:
            resp = future.result()
        except Exception as exc:
            return False, f"{name}: {exc}"
        return bool(resp.success), resp.message or ''

    def call_save_map(self) -> tuple[bool, str]:
        return self._call_trigger(self._save_map_cli, 'save_map', timeout=5.0)

    def call_reset_map(self) -> tuple[bool, str]:
        return self._call_trigger(self._reset_map_cli, 'reset_map', timeout=2.0)

    def call_reload_waypoints(self) -> tuple[bool, str]:
        return self._call_trigger(self._reload_wps_cli, 'reload_waypoints', timeout=2.0)

    # ------------------------------------------------------------------
    # System mode callback
    # ------------------------------------------------------------------

    def _cb_system_mode(self, msg) -> None:
        with self._lock:
            self._state.system_mode = msg.data.strip().upper()

    # ------------------------------------------------------------------
    # ROS2 callbacks (executor thread)
    # ------------------------------------------------------------------

    def _cb_pose(self, msg) -> None:
        import math
        q = msg.pose.orientation
        # Quaternion → yaw (theta)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        with self._lock:
            self._state.pose = {
                "x": round(msg.pose.position.x, 3),
                "y": round(msg.pose.position.y, 3),
                "theta": round(theta, 4),
            }

    def _cb_nav_status(self, msg) -> None:
        with self._lock:
            self._state.nav_status = msg.data

    def _cb_robot_state(self, msg) -> None:
        with self._lock:
            self._state.state = msg.data

    def _cb_sm_blackboard(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        with self._lock:
            self._sm_snapshot = payload

    def _cb_sm_transition(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        with self._lock:
            self._sm_transitions.append(payload)

    def _cb_mission(self, msg) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            data = None
        with self._lock:
            self._state.mission = data

    def _cb_vel(self, msg) -> None:
        with self._lock:
            self._state.velocity = {
                "linear": round(msg.linear.x, 3),
                "angular": round(msg.angular.z, 3),
            }

    def _cb_lifter(self, msg) -> None:
        with self._lock:
            self._state.lifter_level = int(msg.data)

    def _cb_image(self, msg) -> None:
        try:
            jpeg = self._frame_to_jpeg(msg)
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warning(f"Image conversion failed: {exc}")
            return
        with self._lock:
            self._latest_frame = jpeg

    def _cb_map(self, msg) -> None:
        try:
            png = self._map_to_png(msg)
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warning(f"Map conversion failed: {exc}")
            return
        origin = msg.info.origin.position
        with self._lock:
            self._latest_map = png
            self._map_info = {
                "origin_x": round(float(origin.x), 4),
                "origin_y": round(float(origin.y), 4),
                "resolution": round(float(msg.info.resolution), 6),
                "width": int(msg.info.width),
                "height": int(msg.info.height),
                "source": "live",
            }

    def _cb_scan(self, msg) -> None:
        """Downsample every 4th ray and store scan data."""
        ranges_raw = list(msg.ranges)
        # Downsample: keep every 4th ray
        ranges_ds = ranges_raw[::4]
        with self._lock:
            self._state.scan = {
                "ranges": ranges_ds,
                "angle_min": msg.angle_min,
                "angle_increment": msg.angle_increment * 4,
                "range_max": msg.range_max,
            }

    def _cb_plan(self, msg) -> None:
        """Store nav plan as list of {x, y} dicts."""
        points = [
            {"x": round(pose.pose.position.x, 3), "y": round(pose.pose.position.y, 3)}
            for pose in msg.poses
        ]
        with self._lock:
            self._state.nav_plan = points

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _frame_to_jpeg(self, msg) -> bytes:
        """Convert sensor_msgs/Image to JPEG bytes (no cv_bridge dependency)."""
        encoding = msg.encoding.lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)
        h, w = msg.height, msg.width

        if encoding in ("rgb8", "bgr8"):
            img = data.reshape((h, w, 3))
            if encoding == "bgr8":
                img = img[:, :, ::-1]  # BGR → RGB
        elif encoding in ("mono8", "8uc1"):
            img = np.stack([data.reshape((h, w))] * 3, axis=-1)
        elif encoding == "rgba8":
            img = data.reshape((h, w, 4))[:, :, :3]
        elif encoding == "bgra8":
            img = data.reshape((h, w, 4))[:, :, 2::-1]
        else:
            # Fallback: treat raw data as grayscale
            img = np.zeros((h, w, 3), dtype=np.uint8)

        return self._numpy_rgb_to_jpeg(img)

    def _numpy_rgb_to_jpeg(self, rgb: np.ndarray, quality: int = 75) -> bytes:
        """Encode an H×W×3 uint8 NumPy array to JPEG using only stdlib + numpy."""
        try:
            # Prefer OpenCV when available (fast, always present on robot)
            import cv2
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return bytes(buf)
        except ImportError:
            pass
        # PIL fallback
        try:
            from PIL import Image as PILImage
            buf = io.BytesIO()
            PILImage.fromarray(rgb, "RGB").save(buf, format="JPEG", quality=quality)
            return buf.getvalue()
        except ImportError:
            pass
        # Last resort: return a minimal valid JPEG (1×1 black pixel)
        return self._minimal_jpeg()

    @staticmethod
    def _minimal_jpeg() -> bytes:
        """Return a 1×1 black JPEG for when no encoder is available."""
        return bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
            b"\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f"
            b"\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08"
            b"\x01\x01\x00\x00?\x00\xf5\x0a\xff\xd9"
        )

    def _load_static_map(self) -> None:
        """Load pre-saved map from ~/ros2_maps/ as fallback when /map isn't published."""
        import glob
        import yaml

        maps_dir = os.path.expanduser("~/ros2_maps")
        yaml_files = glob.glob(os.path.join(maps_dir, "*.yaml"))
        # Filter out waypoints.yaml
        yaml_files = [p for p in yaml_files if "waypoint" not in os.path.basename(p).lower()]
        if not yaml_files:
            self._node.get_logger().info("No static map YAML found in ~/ros2_maps/")
            return

        # Prefer map.yaml, otherwise take the first matching file
        yaml_path = next(
            (p for p in yaml_files if os.path.basename(p) == "map.yaml"),
            yaml_files[0],
        )

        try:
            with open(yaml_path, "r") as f:
                meta = yaml.safe_load(f)
        except Exception as exc:
            self._node.get_logger().warning(f"Could not read map YAML {yaml_path}: {exc}")
            return

        pgm_file = meta.get("image", "")
        if not pgm_file:
            self._node.get_logger().warning(f"No 'image' field in {yaml_path}")
            return
        if not os.path.isabs(pgm_file):
            pgm_file = os.path.join(maps_dir, pgm_file)

        try:
            gray = self._load_pgm(pgm_file)
        except Exception as exc:
            self._node.get_logger().warning(f"Could not load PGM {pgm_file}: {exc}")
            return

        # PGM from map_saver: row 0 = top of image (highest Y world coord) — no flip needed.
        png = self._numpy_gray_to_png(gray)

        origin = meta.get("origin", [-10.0, -10.0, 0.0])
        resolution = float(meta.get("resolution", 0.05))
        h, w = gray.shape

        with self._lock:
            self._latest_map = png
            self._map_info = {
                "origin_x": float(origin[0]) if isinstance(origin, list) else -10.0,
                "origin_y": float(origin[1]) if isinstance(origin, list) else -10.0,
                "resolution": resolution,
                "width": w,
                "height": h,
                "source": "static",
            }

        self._node.get_logger().info(
            f"Static map loaded: {w}×{h} px, res={resolution:.4f} m/px, "
            f"origin=({self._map_info['origin_x']:.1f}, {self._map_info['origin_y']:.1f}) "
            f"[{os.path.basename(pgm_file)}]"
        )

    @staticmethod
    def _load_pgm(path: str) -> np.ndarray:
        """Load a PGM file and return a uint8 grayscale numpy array."""
        try:
            from PIL import Image as PILImage
            return np.array(PILImage.open(path).convert("L"), dtype=np.uint8)
        except ImportError:
            pass
        try:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
        except ImportError:
            pass
        # Pure-Python P5/P2 PGM reader
        with open(path, "rb") as f:
            magic = f.readline().strip()
            if magic not in (b"P5", b"P2"):
                raise ValueError(f"Not a PGM file: {path}")
            line = f.readline()
            while line.startswith(b"#"):
                line = f.readline()
            w, h = map(int, line.split())
            maxval = int(f.readline().strip())
            if magic == b"P5":
                raw = f.read()
                arr = np.frombuffer(raw, dtype=np.uint8 if maxval < 256 else ">u2")
                if maxval != 255:
                    arr = (arr.astype(np.uint32) * 255 // maxval).astype(np.uint8)
            else:
                values = list(map(int, f.read().split()))
                arr = np.array(values, dtype=np.uint32)
                arr = (arr * 255 // maxval).astype(np.uint8)
        return arr.reshape((h, w))

    def _map_to_png(self, msg) -> bytes:
        """Convert nav_msgs/OccupancyGrid to a PNG bytes image."""
        w, h = msg.info.width, msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((h, w))

        # OccupancyGrid semantics:
        #  -1  → unknown  → gray (128)
        #   0  → free     → white (255)
        # 100  → occupied → black (0)
        img = np.full((h, w), 128, dtype=np.uint8)
        img[data == 0] = 255
        img[data == 100] = 0
        # Partial occupancy (1-99) → scale
        mask = (data > 0) & (data < 100)
        img[mask] = (255 - data[mask].astype(np.int16) * 255 // 100).astype(np.uint8)

        # Flip vertically so Y-up ROS → Y-down image
        img = np.flipud(img)

        return self._numpy_gray_to_png(img)

    @staticmethod
    def _numpy_gray_to_png(gray: np.ndarray) -> bytes:
        """Encode a grayscale H×W uint8 array to PNG bytes using only stdlib."""
        try:
            import cv2
            _, buf = cv2.imencode(".png", gray)
            return bytes(buf)
        except ImportError:
            pass
        try:
            from PIL import Image as PILImage
            buf = io.BytesIO()
            PILImage.fromarray(gray, "L").save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            pass
        # Pure-Python PNG encoder (grayscale, no filtering)
        return _encode_png_gray(gray)


# ---------------------------------------------------------------------------
# Pure-Python minimal PNG encoder (grayscale 8-bit, no compression filters)
# Only used when neither OpenCV nor Pillow is available.
# ---------------------------------------------------------------------------

def _encode_png_gray(gray: np.ndarray) -> bytes:
    """Encode a H×W uint8 grayscale array to a valid PNG byte string."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return length + tag + data + crc

    h, w = gray.shape
    # Build raw image data (filter byte 0 = None before each row)
    raw = b"".join(b"\x00" + bytes(gray[y]) for y in range(h))
    compressed = zlib.compress(raw, 6)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    return png
