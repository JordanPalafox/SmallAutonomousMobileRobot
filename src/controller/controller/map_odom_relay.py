#!/usr/bin/env python3
"""map_odom_relay — reconstruct the map->odom TF on an off-board (laptop) host.

On the distributed real-robot stack the Jetson's slam_node broadcasts map->odom
on /tf, but that particular transform does not match reliably across the WiFi
(FastDDS multi-writer /tf quirk: /tf carries writers from slam_node, real_odom
and robot_state_publisher; the laptop matches real_odom's odom->base_footprint
but often not slam_node's map->odom, even though slam's *other* topics —
/map, /slam_pose, /particle_cloud — all arrive fine).

So instead of relying on that transform crossing, we rebuild it locally from
two things that DO arrive: /slam_pose (map->base_footprint) and the
odom->base_footprint TF (from real_odom):

    map->odom = (map->base_footprint) o (odom->base_footprint)^-1

Pure 2D planar math, no external deps. Runs on the laptop via laptop.launch.py.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster


def _yaw(z, w):
    return 2.0 * math.atan2(z, w)


def _wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class MapOdomRelay(Node):
    def __init__(self):
        super().__init__('map_odom_relay')
        self._buf = Buffer()
        self._listener = TransformListener(self._buf, self)
        self._br = TransformBroadcaster(self)
        self._odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self._base_frame = self.declare_parameter('base_frame', 'base_footprint').value
        self._pose_topic = self.declare_parameter('pose_topic', '/slam_pose').value
        # Exponential-moving-average smoothing on the reconstructed map→odom.
        # /slam_pose carries the RAW AMCL estimate, which steps every scan on
        # particle resampling, scan-to-map refine (±8°) and loop closure.
        # Those steps reach any consumer that reads map→base from TF — e.g.
        # nav_node steering on a smooth heading (pose_source:=tf).  alpha=1.0
        # (default) is pass-through (no behaviour change); set 0.4–0.6 to
        # reject the jitter the same way slam_node's own tf_alpha does for its
        # on-board map→odom.  The 50 Hz odom→base still fills motion between
        # updates, so the lag lands only on the slow correction term.
        self._alpha = float(self.declare_parameter('tf_smoothing_alpha', 1.0).value)
        self._rx = None
        self._ry = None
        self._rth = None
        self.create_subscription(PoseStamped, self._pose_topic, self._on_pose, 10)
        self._warned = False
        self.get_logger().info(
            f'map_odom_relay: map->{self._odom_frame} from {self._pose_topic} '
            f'+ {self._odom_frame}->{self._base_frame}')

    def _on_pose(self, msg):
        mx, my = msg.pose.position.x, msg.pose.position.y
        mth = _yaw(msg.pose.orientation.z, msg.pose.orientation.w)
        try:
            t = self._buf.lookup_transform(self._odom_frame, self._base_frame, Time())
        except Exception:
            if not self._warned:
                self.get_logger().warn(
                    f'waiting for {self._odom_frame}->{self._base_frame} TF...')
                self._warned = True
            return
        ox, oy = t.transform.translation.x, t.transform.translation.y
        oth = _yaw(t.transform.rotation.z, t.transform.rotation.w)

        # base->odom = (odom->base)^-1
        c, s = math.cos(oth), math.sin(oth)
        bx, by, bth = (-ox * c - oy * s), (ox * s - oy * c), -oth
        # map->odom = (map->base) o (base->odom)
        cm, sm = math.cos(mth), math.sin(mth)
        rx = mx + bx * cm - by * sm
        ry = my + bx * sm + by * cm
        rth = mth + bth

        # EMA-smooth the reconstructed map→odom (alpha < 1.0 enables it).
        # Wrap the heading delta so smoothing stays correct across ±π.
        a = self._alpha
        if a < 1.0 and self._rth is not None:
            self._rx = self._rx + a * (rx - self._rx)
            self._ry = self._ry + a * (ry - self._ry)
            self._rth = _wrap(self._rth + a * _wrap(rth - self._rth))
        else:
            self._rx, self._ry, self._rth = rx, ry, rth
        rx, ry, rth = self._rx, self._ry, self._rth

        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = 'map'
        tf.child_frame_id = self._odom_frame
        tf.transform.translation.x = rx
        tf.transform.translation.y = ry
        tf.transform.rotation.z = math.sin(rth / 2.0)
        tf.transform.rotation.w = math.cos(rth / 2.0)
        self._br.sendTransform(tf)


def main():
    rclpy.init()
    try:
        rclpy.spin(MapOdomRelay())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
