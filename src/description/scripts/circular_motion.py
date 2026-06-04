#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class CircularMotion(Node):
    def __init__(self):
        super().__init__('circular_motion')

        self.tf_broadcaster = TransformBroadcaster(self)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Circle parameters
        self.radius = 0.5        # metres
        self.omega = 0.5         # rad/s — angular velocity around the circle
        self.wheel_radius = 0.05 # metres

        # Linear speed of robot = radius * omega
        # Wheel spin rate = linear_speed / wheel_radius
        self.wheel_rate = (self.radius * self.omega) / self.wheel_radius

        self.t = 0.0
        self.dt = 0.033  # ~30 Hz

        self.timer = self.create_timer(self.dt, self.update)

    def update(self):
        self.t += self.dt

        # Robot position on the circle
        x = self.radius * math.cos(self.omega * self.t)
        y = self.radius * math.sin(self.omega * self.t)
        # Robot yaw: tangent to circle (perpendicular to radius vector)
        yaw = self.omega * self.t + math.pi / 2.0

        # --- TF: odom -> base_footprint ---
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_footprint'
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = 0.0
        # Convert yaw to quaternion (rotation around Z)
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = math.sin(yaw / 2.0)
        tf_msg.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf_broadcaster.sendTransform(tf_msg)

        # --- JointState: spinning wheels ---
        wheel_angle = self.wheel_rate * self.t

        js = JointState()
        js.header.stamp = tf_msg.header.stamp
        js.name = ['wheel_right_joint', 'wheel_left_joint']
        js.position = [wheel_angle, wheel_angle]
        js.velocity = [self.wheel_rate, self.wheel_rate]
        self.joint_pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = CircularMotion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
