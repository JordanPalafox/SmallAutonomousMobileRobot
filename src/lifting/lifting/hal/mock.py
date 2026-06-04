"""Mock GPIO driver for development and testing on non-Jetson machines.

All GPIO operations are logged instead of sent to real hardware.
"""

import logging

from lifting.hal.base import GpioDriver

logger = logging.getLogger(__name__)


class MockGpioDriver(GpioDriver):
    """Simulated GPIO driver — logs level changes, no real hardware access."""

    def __init__(self) -> None:
        self._current_level: int = 0

    def setup(self) -> None:
        """Log initialization; no real hardware touched."""
        logger.info('Mock GPIO initialized (no real hardware)')

    def set_level(self, level: int) -> None:
        """Log the requested level in decimal and binary.

        Args:
            level: Integer in [0, 7].
        """
        self._current_level = level
        logger.info('Lifter level: %d  (binary: %s)', level, format(level, '03b'))

    def cleanup(self) -> None:
        """Log cleanup; no real hardware to release."""
        logger.info('Mock GPIO cleaned up')

    @property
    def current_level(self) -> int:
        """Return the most recently set level."""
        return self._current_level
