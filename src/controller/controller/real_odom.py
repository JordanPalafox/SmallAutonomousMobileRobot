#!/usr/bin/env python3
"""
real_odom — wheel-velocity odometry for the real Puzzlebot MCR2.

Subscribes to /wl and /wr (Float32, rad/s) published by velocity_bridge,
integrates them with a fixed-rate timer, and publishes:
  /puzzlebot_controller/odom  (nav_msgs/Odometry)
  TF: odom → base_footprint

This is the real-hardware equivalent of simple_controller's jointCallback,
which uses joint_states positions from ros2_control (sim only).
"""
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def _quat_from_yaw(yaw):
    """Return (x, y, z, w) quaternion for a pure Z rotation."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (0.0, 0.0, sy, cy)


class RealOdom(Node):

    def __init__(self):
        super().__init__('real_odom')
        self.declare_parameter('wheel_radius',    0.05)
        self.declare_parameter('wheel_separation', 0.19)
        self.declare_parameter('odom_rate',       50.0)   # Hz

        r = self.get_parameter('wheel_radius').value
        L = self.get_parameter('wheel_separation').value
        rate = self.get_parameter('odom_rate').value

        self._r = r
        self._L = L

        self._wl = 0.0
        self._wr = 0.0
        self._x  = 0.0
        self._y  = 0.0
        self._th = 0.0

        self._odom_pub = self.create_publisher(Odometry, '/puzzlebot_controller/odom', 10)
        self._tf_br    = TransformBroadcaster(self)

        self._odom_msg = Odometry()
        self._odom_msg.header.frame_id       = 'odom'
        self._odom_msg.child_frame_id        = 'base_footprint'
        self._odom_msg.pose.pose.orientation.w = 1.0

        self._tf_msg = TransformStamped()
        self._tf_msg.header.frame_id = 'odom'
        self._tf_msg.child_frame_id  = 'base_footprint'

        self.create_subscription(Float32, '/wl', lambda m: self._set_wl(m.data), 10)
        self.create_subscription(Float32, '/wr', lambda m: self._set_wr(m.data), 10)

        dt = 1.0 / rate
        self._dt = dt
        self.create_timer(dt, self._update)

        self.get_logger().info(
            f'RealOdom ready — r={r} m  L={L} m  rate={rate} Hz')

    def _set_wl(self, v): self._wl = v
    def _set_wr(self, v): self._wr = v

    def _update(self):
        dt  = self._dt
        wl  = self._wl
        wr  = self._wr

        # forward kinematics
        v     = self._r * (wr + wl) / 2.0
        omega = self._r * (wr - wl) / self._L

        d_s     = v * dt
        d_theta = omega * dt

        self._th += d_theta
        self._x  += d_s * math.cos(self._th)
        self._y  += d_s * math.sin(self._th)

        q   = _quat_from_yaw(self._th)
        now = self.get_clock().now().to_msg()

        self._odom_msg.header.stamp = now
        self._odom_msg.pose.pose.position.x    = self._x
        self._odom_msg.pose.pose.position.y    = self._y
        self._odom_msg.pose.pose.orientation.x = q[0]
        self._odom_msg.pose.pose.orientation.y = q[1]
        self._odom_msg.pose.pose.orientation.z = q[2]
        self._odom_msg.pose.pose.orientation.w = q[3]
        self._odom_msg.twist.twist.linear.x    = v
        self._odom_msg.twist.twist.angular.z   = omega
        self._odom_pub.publish(self._odom_msg)

        self._tf_msg.header.stamp = now
        self._tf_msg.transform.translation.x = self._x
        self._tf_msg.transform.translation.y = self._y
        self._tf_msg.transform.rotation.x    = q[0]
        self._tf_msg.transform.rotation.y    = q[1]
        self._tf_msg.transform.rotation.z    = q[2]
        self._tf_msg.transform.rotation.w    = q[3]
        self._tf_br.sendTransform(self._tf_msg)


def main():
    rclpy.init()
    node = RealOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
