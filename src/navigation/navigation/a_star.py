"""
a_star.py — Pure-function A* path planner (no ROS, no scipy).

Public API
----------
inflate(grid, r_cells) -> np.ndarray
    Circular obstacle dilation using numpy only.

plan(grid, origin_x, origin_y, resolution,
     start_x, start_y, goal_x, goal_y,
     inflation_radius=0.30) -> list[(x,y)] | None
    Full pipeline: inflate → A* → smooth → world coords.

Internal helpers (exported for testing):
    _astar(grid, start_cell, goal_cell) -> list[cells] | None
    _smooth(cells, grid) -> list[cells]
    _has_los(c0, c1, grid) -> bool

Coordinate convention
---------------------
All cell coordinates are (cx, cy) tuples where:
    cx = column index  (x-axis)
    cy = row    index  (y-axis)
Grid indexing: grid[cy, cx]  (row, col)
"""

from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

# Type aliases
Cell = Tuple[int, int]          # (cx, cy)
WorldPt = Tuple[float, float]   # (x, y) metres


# ---------------------------------------------------------------------------
# Public: obstacle inflation
# ---------------------------------------------------------------------------

def inflate(grid: np.ndarray, r_cells: int) -> np.ndarray:
    """Return a copy of *grid* with occupied cells dilated by *r_cells*.

    Only numpy is used — no scipy.  For each occupied cell, every cell within
    Euclidean distance *r_cells* is also marked occupied (100).

    Parameters
    ----------
    grid : np.ndarray, shape (H, W), int8
        Input occupancy grid.  Occupied cells have value 100.
    r_cells : int
        Inflation radius in cells.

    Returns
    -------
    np.ndarray, shape (H, W), int8
        Inflated grid.
    """
    if r_cells <= 0:
        return grid.copy()

    inflated = grid.copy().astype(np.int16)
    H, W = grid.shape

    # Pre-compute relative offsets that fall within a circular disk
    offsets: list[tuple[int, int]] = []
    for dr in range(-r_cells, r_cells + 1):
        for dc in range(-r_cells, r_cells + 1):
            if math.sqrt(dr * dr + dc * dc) <= r_cells:
                offsets.append((dr, dc))

    # Find all occupied cells
    occ_rows, occ_cols = np.where(grid == 100)

    for cy, cx in zip(occ_rows.tolist(), occ_cols.tolist()):
        for dr, dc in offsets:
            ny, nx = cy + dr, cx + dc
            if 0 <= ny < H and 0 <= nx < W:
                inflated[ny, nx] = 100

    return inflated.astype(np.int8)


# ---------------------------------------------------------------------------
# Public: full planning pipeline
# ---------------------------------------------------------------------------

def plan(
    grid: np.ndarray,
    origin_x: float,
    origin_y: float,
    resolution: float,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    inflation_radius: float = 0.30,
) -> Optional[List[WorldPt]]:
    """Plan a collision-free path from (start_x, start_y) to (goal_x, goal_y).

    Parameters
    ----------
    grid : np.ndarray, shape (H, W), int8
        Raw occupancy grid (0=free, 100=occupied, -1=unknown).
    origin_x, origin_y : float
        World coordinates of cell (row=0, col=0) — i.e. bottom-left corner.
    resolution : float
        Metres per cell.
    start_x, start_y, goal_x, goal_y : float
        World-frame coordinates.
    inflation_radius : float
        Obstacle inflation radius in metres.

    Returns
    -------
    list of (x, y) world-frame waypoints, or None if unreachable.
    """
    H, W = grid.shape
    r_cells = int(math.ceil(inflation_radius / resolution))
    inflated = inflate(grid, r_cells)

    # Convert world → cell
    def w2c(wx: float, wy: float) -> Cell:
        cx = int((wx - origin_x) / resolution)
        cy = int((wy - origin_y) / resolution)
        cx = max(0, min(W - 1, cx))
        cy = max(0, min(H - 1, cy))
        return (cx, cy)

    def c2w(cx: int, cy: int) -> WorldPt:
        x = origin_x + (cx + 0.5) * resolution
        y = origin_y + (cy + 0.5) * resolution
        return (x, y)

    start_cell = w2c(start_x, start_y)
    goal_cell = w2c(goal_x, goal_y)

    # Snap start/goal to nearest free cell if inside obstacle
    start_cell = _nearest_free(inflated, start_cell)
    goal_cell = _nearest_free(inflated, goal_cell)
    if start_cell is None or goal_cell is None:
        return None

    cells = _astar(inflated, start_cell, goal_cell)
    if cells is None:
        return None

    cells = _smooth(cells, inflated)
    return [c2w(cx, cy) for cx, cy in cells]


# ---------------------------------------------------------------------------
# Internal: A* search
# ---------------------------------------------------------------------------

def _astar(
    grid: np.ndarray,
    start: Cell,
    goal: Cell,
) -> Optional[List[Cell]]:
    """8-connected A* on *grid*.

    Parameters
    ----------
    grid : np.ndarray, shape (H, W), int8
        Inflated grid.  Only cells where grid[cy, cx] == 100 are blocked.
    start, goal : (cx, cy) tuples.

    Returns
    -------
    list of (cx, cy) cells from start to goal (inclusive), or None.
    """
    H, W = grid.shape
    SQRT2 = math.sqrt(2)

    # 8-connected directions: (dcx, dcy, cost)
    DIRS = [
        (-1, -1, SQRT2), (0, -1, 1.0), (1, -1, SQRT2),
        (-1,  0, 1.0),                  (1,  0, 1.0),
        (-1,  1, SQRT2), (0,  1, 1.0), (1,  1, SQRT2),
    ]

    def h(c: Cell) -> float:
        return math.sqrt((c[0] - goal[0]) ** 2 + (c[1] - goal[1]) ** 2)

    g_score: dict[Cell, float] = {start: 0.0}
    came_from: dict[Cell, Optional[Cell]] = {start: None}

    # heap entry: (f, g, cell)
    heap: list[tuple[float, float, Cell]] = []
    heapq.heappush(heap, (h(start), 0.0, start))

    while heap:
        f, g, current = heapq.heappop(heap)

        if current == goal:
            return _reconstruct(came_from, current)

        if g > g_score.get(current, math.inf):
            continue  # stale entry

        cx, cy = current
        for dcx, dcy, cost in DIRS:
            nx, ny = cx + dcx, cy + dcy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny, nx] == 100:
                continue

            tentative_g = g + cost
            neighbour: Cell = (nx, ny)
            if tentative_g < g_score.get(neighbour, math.inf):
                g_score[neighbour] = tentative_g
                came_from[neighbour] = current
                heapq.heappush(heap, (tentative_g + h(neighbour), tentative_g, neighbour))

    return None


def _reconstruct(
    came_from: dict[Cell, Optional[Cell]],
    current: Cell,
) -> List[Cell]:
    path: List[Cell] = []
    node: Optional[Cell] = current
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Internal: path smoothing (line-of-sight shortcutting)
# ---------------------------------------------------------------------------

def _smooth(cells: List[Cell], grid: np.ndarray) -> List[Cell]:
    """Remove intermediate cells where line-of-sight exists.

    Uses Bresenham line-of-sight checks.  Reduces waypoint count while
    preserving collision-free guarantees.
    """
    if len(cells) <= 2:
        return cells

    result: List[Cell] = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        # Find the furthest cell reachable by LOS from result[-1]
        j = len(cells) - 1
        while j > i + 1:
            if _has_los(result[-1], cells[j], grid):
                break
            j -= 1
        result.append(cells[j])
        i = j

    return result


def _has_los(c0: Cell, c1: Cell, grid: np.ndarray) -> bool:
    """Return True if the Bresenham line from c0 to c1 is obstacle-free."""
    H, W = grid.shape
    x0, y0 = c0
    x1, y1 = c1

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        if not (0 <= x < W and 0 <= y < H):
            return False
        if grid[y, x] == 100:
            return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


# ---------------------------------------------------------------------------
# Internal: nearest free cell (BFS fallback)
# ---------------------------------------------------------------------------

def _nearest_free(grid: np.ndarray, cell: Cell) -> Optional[Cell]:
    """Return the nearest free cell to *cell* via BFS, or None."""
    H, W = grid.shape
    cx, cy = cell
    if 0 <= cx < W and 0 <= cy < H and grid[cy, cx] != 100:
        return cell

    from collections import deque
    visited: set[Cell] = {cell}
    queue: deque[Cell] = deque([cell])

    while queue:
        cx, cy = queue.popleft()
        for dcx, dcy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dcx, cy + dcy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            nc: Cell = (nx, ny)
            if nc in visited:
                continue
            visited.add(nc)
            if grid[ny, nx] != 100:
                return nc
            queue.append(nc)

    return None
