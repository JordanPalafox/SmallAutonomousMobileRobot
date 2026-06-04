"""Odometria de ruedas para Puzzlebot.

Subscribe a /VelocityEncL y /VelocityEncR (Float32, rad/s) e integra
con kinematic diferencial:
    v = r * (wr + wl) / 2
    w = r * (wr - wl) / L
    x += v * cos(th) * dt
    y += v * sin(th) * dt
    th += w * dt

Mantiene la pose (x, y, th) acumulada desde el ultimo reset().
"""
from __future__ import annotations

import math
import time
from typing import Optional

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from std_msgs.msg import Float32


_BE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


class OdometryTracker:
    """Integra encoders del Puzzlebot. Reset() para snapshot del origen."""

    def __init__(
        self,
        node: Node,
        wheel_radius: float = 0.05,
        wheel_separation: float = 0.19,
        topic_left: str = '/VelocityEncL',
        topic_right: str = '/VelocityEncR',
    ) -> None:
        self._r = float(wheel_radius)
        self._L = float(wheel_separation)
        self._wl: float = 0.0
        self._wr: float = 0.0
        self._x: float = 0.0
        self._y: float = 0.0
        self._th: float = 0.0
        self._last_time: Optional[float] = None
        self._got_wl: bool = False
        self._got_wr: bool = False

        node.create_subscription(Float32, topic_left, self._on_wl, _BE_QOS)
        node.create_subscription(Float32, topic_right, self._on_wr, _BE_QOS)

        # Timer interno para integrar a frecuencia fija aunque encoders
        # lleguen a rates distintos.
        node.create_timer(0.02, self._tick)  # 50 Hz

    def _on_wl(self, msg: Float32) -> None:
        self._wl = float(msg.data)
        self._got_wl = True

    def _on_wr(self, msg: Float32) -> None:
        self._wr = float(msg.data)
        self._got_wr = True

    def _tick(self) -> None:
        now = time.monotonic()
        if self._last_time is None:
            self._last_time = now
            return
        dt = now - self._last_time
        self._last_time = now
        if dt <= 0 or dt > 0.5:
            return
        v = self._r * (self._wr + self._wl) * 0.5
        w = self._r * (self._wr - self._wl) / self._L
        # Integracion explicita (Euler)
        d_th = w * dt
        d_s = v * dt
        # Midpoint heading para mejor precision
        th_mid = self._th + 0.5 * d_th
        self._x += d_s * math.cos(th_mid)
        self._y += d_s * math.sin(th_mid)
        self._th = self._th + d_th
        # Wrap [-pi, pi]
        self._th = math.atan2(math.sin(self._th), math.cos(self._th))

    # ------------------------------------------------------------------
    def reset(self, x: float = 0.0, y: float = 0.0, th: float = 0.0) -> None:
        self._x = x
        self._y = y
        self._th = th
        self._last_time = time.monotonic()

    def pose(self) -> tuple[float, float, float]:
        return self._x, self._y, self._th

    def has_data(self) -> bool:
        return self._got_wl and self._got_wr
