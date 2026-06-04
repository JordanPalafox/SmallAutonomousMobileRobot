"""ROS2 node that controls the FPGA lifter via a selectable HAL.

HAL backends
------------
mock    — logs only, no hardware (development).
jetson  — Jetson.GPIO, 3-bit binary on pins 11/13/15 (levels 0-7).
spi     — spidev to Tang Nano 20K, 2-bit level packed in a byte (levels 0-3).

Subscriptions
-------------
/lifter_level  (std_msgs/UInt8)
    Target level. Clamped to the active HAL's range before forwarding.

Publications
------------
/lifter_status (std_msgs/UInt8)
    Current level, published at 1 Hz.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8


class LiftingNode(Node):
    """Bridges /lifter_level commands to hardware via a selectable HAL."""

    def __init__(self) -> None:
        super().__init__('lifting_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('hal', 'mock')  # mock | jetson | spi

        # GPIO (jetson) pinout — kept for backwards compat
        self.declare_parameter('pin_bit0', 11)
        self.declare_parameter('pin_bit1', 13)
        self.declare_parameter('pin_bit2', 15)

        # SPI parameters (Tang Nano 20K)
        self.declare_parameter('spi_bus', 1)
        self.declare_parameter('spi_device', 0)
        self.declare_parameter('spi_speed_hz', 1_000_000)
        self.declare_parameter('spi_mode', 0)

        # Named operational levels (kept for downstream nodes)
        self.declare_parameter('level_rest', 0)
        self.declare_parameter('level_transport', 1)
        self.declare_parameter('level_pick_floor', 3)
        self.declare_parameter('level_carry', 4)
        self.declare_parameter('level_pick_rack2', 5)
        self.declare_parameter('level_max', 7)

        hal: str = self.get_parameter('hal').get_parameter_value().string_value.lower()

        # ── HAL selection ─────────────────────────────────────────────
        if hal == 'spi':
            from lifting.hal.spi import SpiGpioDriver
            self._driver = SpiGpioDriver(
                bus=self.get_parameter('spi_bus').get_parameter_value().integer_value,
                device=self.get_parameter('spi_device').get_parameter_value().integer_value,
                speed_hz=self.get_parameter('spi_speed_hz').get_parameter_value().integer_value,
                mode=self.get_parameter('spi_mode').get_parameter_value().integer_value,
            )
            self._max_level = 3  # 2 bits
            self.get_logger().info('LiftingNode: using SpiGpioDriver (2-bit level)')
        elif hal == 'jetson':
            from lifting.hal.jetson import JetsonGpioDriver
            self._driver = JetsonGpioDriver()
            self._max_level = 7  # 3 bits
            self.get_logger().info('LiftingNode: using JetsonGpioDriver (3-bit level)')
        else:
            from lifting.hal.mock import MockGpioDriver
            self._driver = MockGpioDriver()
            self._max_level = 7
            self.get_logger().info('LiftingNode: using MockGpioDriver')

        self._driver.setup()
        self._current_level: int = 0

        # ── ROS interfaces ────────────────────────────────────────────
        self._sub = self.create_subscription(
            UInt8,
            '/lifter_level',
            self._on_lifter_level,
            10,
        )
        self._status_pub = self.create_publisher(UInt8, '/lifter_status', 10)
        self._status_timer = self.create_timer(1.0, self._publish_status)

        self.get_logger().info(
            f'LiftingNode ready — subscribed to /lifter_level (max level={self._max_level})'
        )

    # ── Callbacks ─────────────────────────────────────────────────────

    def _on_lifter_level(self, msg: UInt8) -> None:
        raw_level = int(msg.data)
        level = max(0, min(self._max_level, raw_level))

        if raw_level != level:
            self.get_logger().warn(
                f'Received out-of-range level {raw_level}; clamped to {level}'
            )

        result = self._driver.set_level(level)
        self._current_level = level

        bits = format(level, '02b' if self._max_level == 3 else '03b')
        if result is not None:
            self.get_logger().info(
                f'Lifter set to {level} ({bits}); FPGA echoed 0x{result:02X}'
            )
        else:
            self.get_logger().info(f'Lifter set to {level} ({bits})')

    def _publish_status(self) -> None:
        msg = UInt8()
        msg.data = self._current_level
        self._status_pub.publish(msg)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self._driver.cleanup()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LiftingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
