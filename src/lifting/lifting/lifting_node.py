"""ROS2 node that controls the FPGA lifter via a selectable HAL.

HAL backends
------------
mock    — logs only, no hardware (development).
jetson  — Jetson.GPIO, 3-bit binary on pins 11/13/15 (levels 0-5).
spi     — spidev to Tang Nano 20K, 3-bit level packed in a byte (levels 0-5).

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
        self.declare_parameter('level_max', 5)

        # File the last commanded level is cached in, so a node RESTART (crash /
        # relaunch) mid-mission restores the fork to where it was instead of
        # dropping it. The Jetson GPIO HAL boots all pins LOW = level 0, so without
        # this a restart while carrying a pallet to the truck would silently lower
        # the load. /tmp is wiped on a full power cycle → a fresh boot still starts
        # at the safe level 0; only an in-session restart restores.
        self.declare_parameter('state_file', '/tmp/puzzlebot_lifter_level')

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
            self._max_level = 5  # 3 bits, FPGA has 6 positions (000-101)
            self.get_logger().info('LiftingNode: using SpiGpioDriver (3-bit level)')
        elif hal == 'jetson':
            from lifting.hal.jetson import JetsonGpioDriver
            self._driver = JetsonGpioDriver()
            self._max_level = 5  # 3 bits, FPGA has 6 positions (000-101)
            self.get_logger().info('LiftingNode: using JetsonGpioDriver (3-bit level)')
        else:
            from lifting.hal.mock import MockGpioDriver
            self._driver = MockGpioDriver()
            self._max_level = 5
            self.get_logger().info('LiftingNode: using MockGpioDriver')

        self._driver.setup()
        self._state_file = self.get_parameter('state_file').get_parameter_value().string_value

        # Restore the last commanded level after a mid-mission restart. setup()
        # has already driven the HAL to its boot default (GPIO → level 0); if a
        # cached level survives we re-drive the FPGA to it so the fork stays put.
        restored = self._load_persisted_level()
        if restored is not None:
            self._driver.set_level(restored)
            self._current_level = restored
            self.get_logger().warn(
                f'Restored lifter to last level {restored} from {self._state_file} '
                f'(node restart — fork held, not dropped).'
            )
        else:
            self._current_level = 0

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
        self._persist_level(level)

        bits = format(level, '03b')
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

    # ── Level persistence (survive a node restart) ─────────────────────

    def _load_persisted_level(self) -> int | None:
        """Read the last commanded level cached on disk. Returns a valid level
        in [0, max] or None if the file is missing / unreadable / out of range."""
        try:
            with open(self._state_file, 'r', encoding='utf-8') as f:
                level = int(f.read().strip())
        except (FileNotFoundError, ValueError, OSError):
            return None
        return level if 0 <= level <= self._max_level else None

    def _persist_level(self, level: int) -> None:
        """Cache the current level so a restart can restore it. Best-effort —
        a write failure is logged but never breaks lifter control."""
        try:
            with open(self._state_file, 'w', encoding='utf-8') as f:
                f.write(str(int(level)))
        except OSError as exc:
            self.get_logger().warn(f'Could not persist lifter level: {exc}')

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
