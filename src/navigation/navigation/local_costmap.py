"""
local_costmap.py — Rolling, world-anchored LiDAR costmap with temporal decay.

The costmap is a square patch of the world centred on (cx, cy).  Each cell
stores a float score in [0, 1]:
    * Scan hits raise the score to 1.0.
    * Each `decay(dt)` call multiplies every cell by exp(-dt / tau).
    * Cells with score >= `occ_threshold` are considered occupied.

Decay makes the costmap forgive transient detections (a person walking past
sheds its trail within ~1 s) while persistent hits remain occupied because
they are refreshed every cycle.

The patch is re-centred on demand via `recenter(cx, cy)`.  Cells that fall
outside the new patch are dropped; new cells start at 0.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from sensor_msgs.msg import LaserScan


class LocalCostmap:
    """World-anchored rolling LiDAR costmap with temporal decay."""

    def __init__(
        self,
        radius_m: float = 2.0,
        resolution: float = 0.05,
        decay_tau: float = 0.7,
        occ_threshold: float = 0.4,
    ) -> None:
        self.radius_m = float(radius_m)
        self.resolution = float(resolution)
        self.decay_tau = float(decay_tau)
        self.occ_threshold = float(occ_threshold)

        # Patch is square: side = 2*radius / res cells
        self._side = max(1, int(round(2.0 * self.radius_m / self.resolution)))
        self._scores: np.ndarray = np.zeros((self._side, self._side), dtype=np.float32)

        # World coords of the cell (row=0, col=0) — bottom-left corner of patch
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0
        self._centered: bool = False

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @property
    def side_cells(self) -> int:
        return self._side

    @property
    def origin(self) -> Tuple[float, float]:
        return (self._origin_x, self._origin_y)

    def world_to_cell(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        if not self._centered:
            return None
        cx = int((wx - self._origin_x) / self.resolution)
        cy = int((wy - self._origin_y) / self.resolution)
        if 0 <= cx < self._side and 0 <= cy < self._side:
            return (cx, cy)
        return None

    def is_occupied_world(self, wx: float, wy: float) -> bool:
        cell = self.world_to_cell(wx, wy)
        if cell is None:
            return False
        cx, cy = cell
        return bool(self._scores[cy, cx] >= self.occ_threshold)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def recenter(self, cx: float, cy: float) -> None:
        """Re-anchor the patch around world point (cx, cy), preserving
        scores for cells that remain inside the new window.
        """
        new_origin_x = cx - self.radius_m
        new_origin_y = cy - self.radius_m

        if not self._centered:
            self._origin_x = new_origin_x
            self._origin_y = new_origin_y
            self._centered = True
            return

        # Integer shift between old and new origins (in cells)
        dx_cells = int(round((self._origin_x - new_origin_x) / self.resolution))
        dy_cells = int(round((self._origin_y - new_origin_y) / self.resolution))

        if dx_cells == 0 and dy_cells == 0:
            return

        new_scores = np.zeros_like(self._scores)

        # For each cell in the new grid, find the corresponding cell in
        # the old grid (if any) and copy its score.  Using vectorised
        # slicing for speed.
        # Source range in old grid that maps to new grid:
        # new[r, c] = old[r - dy_cells, c - dx_cells]
        src_y0 = max(0, -dy_cells)
        src_y1 = min(self._side, self._side - dy_cells)
        src_x0 = max(0, -dx_cells)
        src_x1 = min(self._side, self._side - dx_cells)

        dst_y0 = max(0, dy_cells)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, dx_cells)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if src_y1 > src_y0 and src_x1 > src_x0:
            new_scores[dst_y0:dst_y1, dst_x0:dst_x1] = (
                self._scores[src_y0:src_y1, src_x0:src_x1]
            )

        self._scores = new_scores
        self._origin_x = new_origin_x
        self._origin_y = new_origin_y

    def decay(self, dt: float) -> None:
        """Exponentially decay every cell's score by dt seconds."""
        if dt <= 0.0 or self.decay_tau <= 0.0:
            return
        factor = math.exp(-dt / self.decay_tau)
        self._scores *= factor
        # Clamp very small values to 0 to keep the matrix sparse-ish
        self._scores[self._scores < 1e-3] = 0.0

    def integrate_scan(
        self,
        scan: LaserScan,
        rx: float,
        ry: float,
        rtheta: float,
        lidar_yaw: float = 0.0,
    ) -> None:
        """Mark scan hit cells with score 1.0 (in-place).

        Rays beyond scan.range_max are treated as no information.

        `lidar_yaw` is the yaw of the scan frame relative to the robot body
        (base_link).  It is added to every ray bearing so a rear-mounted
        LiDAR (yaw=π on the real Puzzlebot) projects hits to the correct
        world location instead of 180° away.
        """
        if not self._centered or scan is None:
            return

        angles = np.array(
            [scan.angle_min + i * scan.angle_increment
             for i in range(len(scan.ranges))],
            dtype=np.float32,
        )
        ranges = np.asarray(scan.ranges, dtype=np.float32)

        valid = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
        if not np.any(valid):
            return

        a = angles[valid]
        r = ranges[valid]

        bearing = rtheta + lidar_yaw + a
        wx = rx + r * np.cos(bearing)
        wy = ry + r * np.sin(bearing)

        cx = ((wx - self._origin_x) / self.resolution).astype(np.int32)
        cy = ((wy - self._origin_y) / self.resolution).astype(np.int32)

        inside = (cx >= 0) & (cx < self._side) & (cy >= 0) & (cy < self._side)
        if not np.any(inside):
            return

        self._scores[cy[inside], cx[inside]] = 1.0

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def occupied_mask(self) -> np.ndarray:
        """Boolean mask of cells whose score >= occ_threshold."""
        return self._scores >= self.occ_threshold

    def segment_blocked(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        inflation_cells: int = 0,
    ) -> bool:
        """Bresenham-style raycast through the costmap.

        Returns True if any cell along the segment (x0,y0)→(x1,y1) is
        currently occupied.  `inflation_cells` adds an L∞ neighbourhood
        check around each traversed cell.
        """
        if not self._centered:
            return False

        ax = int((x0 - self._origin_x) / self.resolution)
        ay = int((y0 - self._origin_y) / self.resolution)
        bx = int((x1 - self._origin_x) / self.resolution)
        by = int((y1 - self._origin_y) / self.resolution)

        dx = abs(bx - ax)
        dy = abs(by - ay)
        sx = 1 if bx > ax else -1
        sy = 1 if by > ay else -1
        err = dx - dy

        x, y = ax, ay
        side = self._side
        thr = self.occ_threshold
        scores = self._scores

        while True:
            if 0 <= x < side and 0 <= y < side:
                if inflation_cells <= 0:
                    if scores[y, x] >= thr:
                        return True
                else:
                    x_lo = max(0, x - inflation_cells)
                    x_hi = min(side, x + inflation_cells + 1)
                    y_lo = max(0, y - inflation_cells)
                    y_hi = min(side, y + inflation_cells + 1)
                    if np.any(scores[y_lo:y_hi, x_lo:x_hi] >= thr):
                        return True
            if x == bx and y == by:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
