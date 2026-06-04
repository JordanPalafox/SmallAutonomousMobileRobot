"""
map_io.py — Shared map and waypoint I/O utilities (no ROS dependencies).

Public API
----------
load_map(yaml_path) -> (grid, origin_x, origin_y, resolution)
    Reads a nav2-format PGM+YAML pair.  Grid values follow ROS convention:
        0   = free
        100 = occupied
        -1  = unknown

load_waypoints(yaml_path) -> dict
    Returns {name: {x, y, theta}} from a waypoints YAML file.
"""

from __future__ import annotations

import os
import struct
from typing import Tuple

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

def load_map(
    yaml_path: str,
) -> Tuple[np.ndarray, float, float, float]:
    """Load a nav2-format map from *yaml_path*.

    Parameters
    ----------
    yaml_path:
        Absolute path to the ``.yaml`` map descriptor.

    Returns
    -------
    grid : np.ndarray, shape (height, width), dtype int8
        Occupancy values: 0=free, 100=occupied, -1=unknown.
        Row 0 corresponds to the bottom of the map (ROS convention).
    origin_x, origin_y : float
        World coordinates of the bottom-left corner of the grid.
    resolution : float
        Metres per cell.
    """
    yaml_path = os.path.expanduser(yaml_path)

    with open(yaml_path, 'r') as fh:
        meta = yaml.safe_load(fh)

    resolution: float = float(meta['resolution'])
    origin: list = meta['origin']          # [x, y, yaw]
    origin_x: float = float(origin[0])
    origin_y: float = float(origin[1])

    negate: int = int(meta.get('negate', 0))
    occupied_thresh: float = float(meta.get('occupied_thresh', 0.65))
    free_thresh: float = float(meta.get('free_thresh', 0.196))

    # Resolve PGM path relative to the YAML file
    image_file: str = meta['image']
    if not os.path.isabs(image_file):
        image_file = os.path.join(os.path.dirname(yaml_path), image_file)

    pixels = _read_pgm(image_file)        # shape (h, w), dtype uint8

    # PGM row 0 = top of image; ROS row 0 = bottom → flip vertically
    pixels = np.flipud(pixels)

    # Convert pixel brightness to occupancy probability
    if negate:
        occupancy_prob = pixels.astype(np.float32) / 255.0
    else:
        occupancy_prob = 1.0 - pixels.astype(np.float32) / 255.0

    # Classify into 0 / 100 / -1
    grid = np.full(pixels.shape, -1, dtype=np.int8)
    grid[occupancy_prob < free_thresh] = 0
    grid[occupancy_prob >= occupied_thresh] = 100

    return grid, origin_x, origin_y, resolution


def _read_pgm(path: str) -> np.ndarray:
    """Parse a binary P5 PGM file without PIL/Pillow.

    Returns
    -------
    np.ndarray, shape (height, width), dtype uint8
    """
    with open(path, 'rb') as fh:
        # Read magic number
        magic = fh.readline().strip()
        if magic != b'P5':
            raise ValueError(f'Expected P5 PGM, got magic={magic!r} in {path}')

        # Skip comment lines
        while True:
            line = fh.readline()
            if not line.startswith(b'#'):
                break
        dims = line.split()

        # May need another line if width/height are on separate lines
        while len(dims) < 2:
            dims += fh.readline().split()

        width = int(dims[0])
        height = int(dims[1])

        # Read maxval (might be on same or next line)
        maxval_parts = []
        while not maxval_parts:
            maxval_parts = fh.readline().split()
        maxval = int(maxval_parts[0])

        if maxval > 255:
            raise ValueError(f'16-bit PGM not supported (maxval={maxval})')

        # Read raw pixel data
        raw = fh.read(width * height)
        if len(raw) != width * height:
            raise ValueError(
                f'PGM data length mismatch: expected {width * height}, '
                f'got {len(raw)}'
            )

    pixels = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
    return pixels


# ---------------------------------------------------------------------------
# Waypoint loading
# ---------------------------------------------------------------------------

def load_waypoints(yaml_path: str) -> dict:
    """Load waypoints from *yaml_path*.

    Expected format::

        waypoints:
          truck_1: {x: 1.0, y: 2.0, theta: 0.0}
          rack_A1: {x: 3.5, y: 1.2, theta: 1.5708}

    Returns
    -------
    dict
        ``{name: {x, y, theta}}``
    """
    yaml_path = os.path.expanduser(yaml_path)
    with open(yaml_path, 'r') as fh:
        data = yaml.safe_load(fh)
    waypoints = data.get('waypoints', {}) or {}
    # Normalise: ensure each entry has float values
    result: dict = {}
    for name, wp in waypoints.items():
        result[name] = {
            'x': float(wp.get('x', 0.0)),
            'y': float(wp.get('y', 0.0)),
            'theta': float(wp.get('theta', 0.0)),
        }
    return result
