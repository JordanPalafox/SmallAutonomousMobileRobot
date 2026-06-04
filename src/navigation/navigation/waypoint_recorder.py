"""
Waypoint recorder for the Puzzlebot warehouse AMR.

Workflow:
  1. ros2 launch navigation waypoint_recording.launch.py
  2. In RViz: click the "2D Pose Estimate" button (arrow icon), then
     click+drag on the map to set position and heading for each zone.
  3. In this terminal: type a label (e.g. truck_1, rack_A1, roller_2)
     and press Enter to save it.  Press Enter with no label to discard.
  4. Ctrl+C when done — waypoints are saved automatically.

Output:
  ~/ros2_maps/waypoints.yaml  (override with --ros-args -p output:=<path>)
"""
import math
import os
import queue
import signal
import threading

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField
from pathlib import Path
import struct

# Transient Local so late-joining subscribers (RViz) get the last message
_LATCHED = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


def _yaw_from_q(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _load_pgm_yaml(yaml_path: str):
    """Read a nav2-format map YAML+PGM and return an OccupancyGrid message."""
    yaml_path = os.path.expanduser(yaml_path)
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    pgm_name = meta['image']
    if not os.path.isabs(pgm_name):
        pgm_name = os.path.join(os.path.dirname(yaml_path), pgm_name)

    with open(pgm_name, 'rb') as f:
        raw = f.read()

    # Parse PGM header (P5 binary or P2 ASCII)
    lines = []
    i = 0
    while len(lines) < 3:
        end = raw.index(b'\n', i)
        line = raw[i:end].decode().strip()
        i = end + 1
        if line.startswith('#'):
            continue
        lines.append(line)
    fmt = lines[0]
    w, h = map(int, lines[1].split())
    # lines[2] is max_val
    pixel_data = raw[i:]

    res    = float(meta['resolution'])
    negate = int(meta.get('negate', 0))
    o_th   = float(meta.get('occupied_thresh', 0.65))
    f_th   = float(meta.get('free_thresh', 0.196))
    origin = meta['origin']

    # Convert pixels → int8 occupancy (flip rows: PGM top=row0, ROS bottom=row0)
    if fmt == 'P5':
        pixels = list(pixel_data[:w * h])
    else:  # P2 ASCII
        pixels = list(map(int, pixel_data.split()))

    data = []
    for row in range(h - 1, -1, -1):
        for col in range(w):
            p = pixels[row * w + col]
            occupancy = 1.0 - p / 255.0 if not negate else p / 255.0
            if occupancy > o_th:
                data.append(100)
            elif occupancy < f_th:
                data.append(0)
            else:
                data.append(-1)

    msg = OccupancyGrid()
    msg.info.resolution = res
    msg.info.width  = w
    msg.info.height = h
    msg.info.origin.position.x = float(origin[0])
    msg.info.origin.position.y = float(origin[1])
    msg.info.origin.orientation.w = 1.0
    msg.data = data
    return msg


class WaypointRecorder(Node):

    def __init__(self):
        super().__init__('waypoint_recorder')
        self.declare_parameter('map_yaml',
                               os.path.expanduser('~/ros2_maps/warehouse.yaml'))
        self.declare_parameter('output',
                               os.path.expanduser('~/ros2_maps/waypoints.yaml'))

        self._waypoints: dict = {}
        self._pending: queue.Queue = queue.Queue()

        self._map_pub    = self.create_publisher(OccupancyGrid, '/map', _LATCHED)
        self._cloud_pub  = self.create_publisher(PointCloud2, '/map_walls', _LATCHED)
        self._marker_pub = self.create_publisher(MarkerArray, '/waypoint_markers', _LATCHED)
        self._cached_grid: OccupancyGrid | None = None

        # Accept from EITHER RViz tool:
        #   "2D Pose Estimate" → /initialpose (PoseWithCovarianceStamped)
        #   "2D Goal Pose"     → /goal_pose   (PoseStamped)
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self._initialpose_cb, 10)
        self.create_subscription(
            PoseStamped, '/goal_pose', self._goalpose_cb, 10)

        self._load_existing_waypoints()
        self._load_map_cache()
        # Republish every 3 s so RViz gets it regardless of connection timing
        self.create_timer(3.0, self._republish_map)

        output = self.get_parameter('output').value
        self.get_logger().info(
            f'Waypoint recorder ready\n'
            f'  Map: {self.get_parameter("map_yaml").value}\n'
            f'  Output: {output}\n'
            f'  Loaded {len(self._waypoints)} existing waypoints\n'
            f'  In RViz: use the [2D Goal Pose] or [2D Pose Estimate] button to place waypoints'
        )

    def _load_existing_waypoints(self):
        path = os.path.expanduser(self.get_parameter('output').value)
        if not os.path.isfile(path):
            return
        try:
            with open(path) as f:
                doc = yaml.safe_load(f)
            loaded = doc.get('waypoints', {}) or {}
            self._waypoints.update(loaded)
            self.get_logger().info(f'Loaded {len(loaded)} waypoints from {path}')
        except Exception as exc:
            self.get_logger().warn(f'Could not load existing waypoints: {exc}')

    def _load_map_cache(self):
        yaml_path = self.get_parameter('map_yaml').value
        try:
            self._cached_grid = _load_pgm_yaml(yaml_path)
            self._cached_grid.header.frame_id = 'map'
            self.get_logger().info(
                f'Map loaded: {self._cached_grid.info.width}x{self._cached_grid.info.height} '
                f'@ {self._cached_grid.info.resolution:.3f} m/cell')
            self._republish_map()
        except Exception as exc:
            self.get_logger().error(f'Failed to load map: {exc}')

    def _republish_map(self):
        if self._cached_grid is None:
            return
        now = self.get_clock().now().to_msg()
        self._cached_grid.header.stamp = now
        self._map_pub.publish(self._cached_grid)
        self._cloud_pub.publish(self._grid_to_cloud(self._cached_grid))
        if self._waypoints:
            self._publish_markers()

    def _grid_to_cloud(self, grid: OccupancyGrid) -> PointCloud2:
        """Publish occupied cells as PointCloud2 — avoids the indexed_8bit_image GLSL bug."""
        w, h = grid.info.width, grid.info.height
        res  = grid.info.resolution
        ox   = grid.info.origin.position.x
        oy   = grid.info.origin.position.y
        buf  = bytearray()
        for row in range(h):
            for col in range(w):
                if grid.data[row * w + col] == 100:
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
        return msg

    def _publish_markers(self):
        array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Clear previous markers
        clear = Marker()
        clear.action = Marker.DELETEALL
        clear.header.frame_id = 'map'
        clear.header.stamp = now
        array.markers.append(clear)

        for idx, (label, wp) in enumerate(self._waypoints.items()):
            x, y, theta = wp['x'], wp['y'], wp['theta']
            half = theta / 2.0
            qz = math.sin(half)
            qw = math.cos(half)

            # Arrow showing position + heading
            arrow = Marker()
            arrow.header.frame_id = 'map'
            arrow.header.stamp = now
            arrow.ns = 'wp_arrows'
            arrow.id = idx * 2
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = x
            arrow.pose.position.y = y
            arrow.pose.position.z = 0.05
            arrow.pose.orientation.z = qz
            arrow.pose.orientation.w = qw
            arrow.scale.x = 0.5   # shaft length
            arrow.scale.y = 0.08  # shaft diameter
            arrow.scale.z = 0.08  # head diameter
            arrow.color.r = 0.1
            arrow.color.g = 0.9
            arrow.color.b = 0.3
            arrow.color.a = 1.0
            array.markers.append(arrow)

            # Text label above arrow
            text = Marker()
            text.header.frame_id = 'map'
            text.header.stamp = now
            text.ns = 'wp_labels'
            text.id = idx * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = 0.4
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = label
            array.markers.append(text)

        self._marker_pub.publish(array)

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped):
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        theta = _yaw_from_q(msg.pose.pose.orientation)
        self._pending.put((x, y, theta))

    def _goalpose_cb(self, msg: PoseStamped):
        x     = msg.pose.position.x
        y     = msg.pose.position.y
        theta = _yaw_from_q(msg.pose.orientation)
        self._pending.put((x, y, theta))

    def interactive_loop(self):
        print('\n' + '='*60)
        print('  WAYPOINT RECORDER')
        print('='*60)
        print('  RViz → click [2D Goal Pose] OR [2D Pose Estimate] → click+drag on map')
        print('  Then type a label here and press Enter.')
        print('  Press Enter with no label to discard the point.')
        print('  Ctrl+C to finish and save.')
        print('='*60 + '\n')

        while rclpy.ok():
            try:
                x, y, theta = self._pending.get(timeout=0.5)
                deg = math.degrees(theta)
                prompt = (f'  Received: x={x:.3f}  y={y:.3f}  θ={deg:.1f}°\n'
                          f'  Label (e.g. truck_1, rack_A2, roller_3): ')
                label = input(prompt).strip()
                if label:
                    self._waypoints[label] = {
                        'x':     round(x, 4),
                        'y':     round(y, 4),
                        'theta': round(theta, 4),
                    }
                    self._publish_markers()
                    self.save()
                    print(f'  ✓ Saved "{label}" — arrow visible in RViz\n')
                else:
                    print('  ✗ Discarded\n')
            except queue.Empty:
                continue
            except (KeyboardInterrupt, EOFError):
                break

    def save(self):
        if not self._waypoints:
            self.get_logger().warn('No waypoints to save.')
            return
        path = os.path.expanduser(self.get_parameter('output').value)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        doc = {'waypoints': self._waypoints}
        with open(path, 'w') as f:
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
        self.get_logger().info(
            f'Saved {len(self._waypoints)} waypoints → {path}')
        print(f'\n  Saved {len(self._waypoints)} waypoints to:\n  {path}\n')


def main():
    rclpy.init()
    node = WaypointRecorder()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Save on SIGTERM / SIGHUP too (xterm sends these instead of SIGINT when
    # the launch process shuts down or the window is closed).
    def _handle_exit(signum, frame):
        node.save()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_exit)
    signal.signal(signal.SIGHUP, _handle_exit)

    try:
        node.interactive_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
