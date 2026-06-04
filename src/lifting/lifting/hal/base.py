from __future__ import annotations

from abc import ABC, abstractmethod


class GpioDriver(ABC):
    """Abstract base class for GPIO drivers controlling the FPGA lifter."""

    @abstractmethod
    def setup(self) -> None:
        """Initialize GPIO hardware."""

    @abstractmethod
    def set_level(self, level: int) -> int | None:
        """Set lifter to the given level via the underlying transport.

        Backends:
            jetson — 3-bit binary on pins 11/13/15, levels 0-7.
            spi    — 2-bit level packed in a byte, levels 0-3.
            mock   — logs only.

        Args:
            level: Non-negative integer; range depends on backend.

        Returns:
            For SPI, the byte echoed by the FPGA. Otherwise None.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release GPIO resources."""
