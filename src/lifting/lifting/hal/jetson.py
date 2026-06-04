"""Real Jetson Nano GPIO implementation for the FPGA lifter.

3-bit binary encoding:
    bit 0 → board pin 11
    bit 1 → board pin 13
    bit 2 → board pin 15

Jetson.GPIO is imported at runtime so this module can be imported on
non-Jetson machines without raising ImportError.
"""

from __future__ import annotations

from lifting.hal.base import GpioDriver


class JetsonGpioDriver(GpioDriver):
    """GPIO driver for Jetson Nano hardware using Jetson.GPIO."""

    # Maps bit index to BOARD-mode pin number
    PINS: dict[int, int] = {0: 11, 1: 13, 2: 15}

    def setup(self) -> None:
        """Initialize GPIO pins as outputs, all LOW."""
        import Jetson.GPIO as GPIO  # noqa: N813 — runtime import

        self._GPIO = GPIO
        GPIO.setmode(GPIO.BOARD)
        for pin in self.PINS.values():
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    def set_level(self, level: int) -> None:
        """Drive 3-bit binary level to GPIO pins.

        Args:
            level: Integer in [0, 7].  Values outside this range are
                   clamped by the caller (LiftingNode) before arriving here.
        """
        GPIO = self._GPIO
        for bit, pin in self.PINS.items():
            state = GPIO.HIGH if (level >> bit) & 1 else GPIO.LOW
            GPIO.output(pin, state)

    def cleanup(self) -> None:
        """Release all GPIO resources."""
        self._GPIO.cleanup()
