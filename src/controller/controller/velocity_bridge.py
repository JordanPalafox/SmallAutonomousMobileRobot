"""
velocity_bridge.py — bridge Puzzlebot encoder topics to standard wheel-vel.

    /VelocityEncL → /wl   (left  wheel, rad/s, Float32)
    /VelocityEncR → /wr   (right wheel, rad/s, Float32)

The hackerboard firmware publishes encoder readings on the prefixed
topics; the rest of the stack (real_odom, downstream tools) consumes
the canonical /wl /wr.  Pure relay — no transformation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32


class VelocityBridge(Node):

    def __init__(self) -> None:
        super().__init__('velocity_bridge')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._wl_pub = self.create_publisher(Float32, '/wl', 10)
        self._wr_pub = self.create_publisher(Float32, '/wr', 10)
        self.create_subscription(
            Float32, '/VelocityEncL',
            lambda m: self._wl_pub.publish(m), qos)
        self.create_subscription(
            Float32, '/VelocityEncR',
            lambda m: self._wr_pub.publish(m), qos)
        self.get_logger().info(
            'Velocity Bridge started — /VelocityEncL → /wl | '
            '/VelocityEncR → /wr')


def main():
    rclpy.init()
    node = VelocityBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
