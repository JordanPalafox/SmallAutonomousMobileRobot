"""
velocity_bridge.py — bridge Puzzlebot encoder topics to standard wheel-vel.

    /VelocityEncL → /wl   (left  wheel, rad/s, Float32)
    /VelocityEncR → /wr   (right wheel, rad/s, Float32)

The hackerboard firmware publishes the encoder readings on the prefixed topics
at a HIGH rate (hundreds of Hz). Relaying every message 1:1 made this node — a
pure relay — eat ~30% CPU on the 2 GB Nano, and flooded /wl,/wr to EVERY
downstream subscriber (real_odom, vel_smoother's stall guard, the laptop SM) plus
the WiFi link.

So instead of republishing on every incoming message, we keep only the LATEST
reading and republish it at a fixed ``relay_rate`` (default 50 Hz) on a timer.
Downstream only ever needs ~50 Hz: real_odom integrates at 50 Hz (it just samples
the latest /wl,/wr each tick), the stall guard checks at 50 Hz, and the SM's
wheel_speed/lin_vel are fine at 50 Hz. Capping /wl,/wr at 50 Hz cuts this node's
publish work, the downstream callback load, and the WiFi traffic — with no loss
for any consumer. Set ``relay_rate`` <= 0 to fall back to the old 1:1 passthrough.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32


class VelocityBridge(Node):

    def __init__(self) -> None:
        super().__init__('velocity_bridge')
        # Output rate for /wl,/wr (Hz). <=0 ⇒ legacy 1:1 passthrough.
        self.declare_parameter('relay_rate', 50.0)
        rate = float(self.get_parameter('relay_rate').value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._wl_pub = self.create_publisher(Float32, '/wl', 10)
        self._wr_pub = self.create_publisher(Float32, '/wr', 10)

        if rate > 0.0:
            # Throttled: store the latest encoder reading; a timer republishes it
            # at `rate` Hz so the high-rate firmware stream doesn't peg the CPU.
            self._wl_msg = None
            self._wr_msg = None
            self.create_subscription(Float32, '/VelocityEncL', self._on_wl, qos)
            self.create_subscription(Float32, '/VelocityEncR', self._on_wr, qos)
            self.create_timer(1.0 / rate, self._republish)
            self.get_logger().info(
                f'Velocity Bridge started (throttled to {rate:.0f} Hz) — '
                '/VelocityEncL → /wl | /VelocityEncR → /wr')
        else:
            # Legacy 1:1 passthrough (relay every incoming message).
            self.create_subscription(
                Float32, '/VelocityEncL', lambda m: self._wl_pub.publish(m), qos)
            self.create_subscription(
                Float32, '/VelocityEncR', lambda m: self._wr_pub.publish(m), qos)
            self.get_logger().info(
                'Velocity Bridge started (1:1 passthrough) — '
                '/VelocityEncL → /wl | /VelocityEncR → /wr')

    def _on_wl(self, m: Float32) -> None:
        self._wl_msg = m

    def _on_wr(self, m: Float32) -> None:
        self._wr_msg = m

    def _republish(self) -> None:
        if self._wl_msg is not None:
            self._wl_pub.publish(self._wl_msg)
        if self._wr_msg is not None:
            self._wr_pub.publish(self._wr_msg)


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
