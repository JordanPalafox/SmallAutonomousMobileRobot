"""SPI driver for the FPGA lifter (Jetson Nano -> Tang Nano 20K).

Sends a single byte per command over SPI. The lower 2 bits encode the
lifter level (00, 01, 10, 11). spidev is imported at runtime so this
module can be imported on machines without /dev/spidev*.
"""

from __future__ import annotations

from lifting.hal.base import GpioDriver


class SpiGpioDriver(GpioDriver):
    """SPI driver — sends 2-bit level to a Tang Nano 20K FPGA via spidev."""

    def __init__(
        self,
        bus: int = 1,
        device: int = 0,
        speed_hz: int = 1_000_000,
        mode: int = 0b00,
    ) -> None:
        self._bus = bus
        self._device = device
        self._speed_hz = speed_hz
        self._mode = mode
        self._spi = None

    def setup(self) -> None:
        """Open /dev/spidevB.D and configure speed + mode."""
        import spidev  # runtime import

        self._spi = spidev.SpiDev()
        self._spi.open(self._bus, self._device)
        self._spi.max_speed_hz = self._speed_hz
        self._spi.mode = self._mode

    def set_level(self, level: int) -> int:
        """Send the 2-bit level over SPI and return the byte echoed by the FPGA.

        Args:
            level: Integer in [0, 3]. Clamping is the caller's responsibility,
                but the driver also masks to 2 bits as a safety net.
        """
        tx = level & 0x03
        rx = self._spi.xfer2([tx])
        return rx[0]

    def cleanup(self) -> None:
        """Close the SPI device."""
        if self._spi is not None:
            self._spi.close()
            self._spi = None
