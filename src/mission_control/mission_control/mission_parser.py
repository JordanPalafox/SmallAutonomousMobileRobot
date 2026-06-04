"""
mission_parser.py
-----------------
Parse and validate /mission JSON payloads.

Three mission types are supported:

    ROLLER_TO_TRUCK
        Search a list of roller waypoints sequentially, read the QR on the
        pallet at each one, deliver the matching pallet to the truck whose
        alias the QR encodes.

    RACK_TO_TRUCK
        Same idea against rack waypoints. Pickup level depends on whether
        the candidate is a rack_l1 or rack_l2 entry in zones.yaml.

    CUSTOM
        Hand-crafted single-leg mission: explicit source waypoint, explicit
        destination, optional QR scan. Useful for debugging and one-off
        deliveries.

The parser stays pure-Python (no ROS) so it is unit-testable. It needs the
zones registry loaded from zones.yaml; pass it explicitly to keep this module
free of filesystem side effects.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Optional

import yaml


# Mission JSON envelope: {"id", "type", ...}
#
# SEARCH_ROLLERS / SEARCH_RACKS are "search only": drive the zone's candidates
# looking for a pallet QR and STOP once one is found (no pick / nav / place).
# They reuse the ROLLER/RACK search builder with search_only=True. Triggered by
# the voice words "rollers" / "racks".
MISSION_TYPES = (
    "ROLLER_TO_TRUCK", "RACK_TO_TRUCK", "CUSTOM", "PICK_ONLY",
    "SEARCH_ROLLERS", "SEARCH_RACKS",
)

# Lifter levels per zone class. Override per mission via pickup_level/place_level.
# Racks are a single class now (the level is decided at mission time, not baked
# into the waypoint); the SPI HAL only reaches level 3 anyway.
DEFAULT_PICKUP_LEVELS = {
    "rollers": 1,
    "racks":   3,
    "trucks":  2,
}
DEFAULT_PLACE_LEVEL_TRUCK = 2

# Map a waypoint-name prefix to its zone class. The dashboard auto-names
# waypoints roller_N / rack_N / truck_N, so the class is derivable from the name
# — no manual zone lists to keep in sync.
_PREFIX_TO_CLASS = (
    ("roller", "rollers"),
    ("rack",   "racks"),
    ("truck",  "trucks"),
)


def derive_zones(waypoint_names) -> dict:
    """Group waypoint names into zone classes by name prefix.

    Returns {'rollers': [...], 'racks': [...], 'trucks': [...]} sorted by name.
    """
    zones: dict = {"rollers": [], "racks": [], "trucks": []}
    for name in sorted(waypoint_names):
        for prefix, cls in _PREFIX_TO_CLASS:
            if name.startswith(prefix):
                zones[cls].append(name)
                break
    return zones


def load_zones(yaml_path: str) -> dict:
    """Load zones.yaml — only ``qr_aliases`` is required now.

    Zone classification is derived from waypoint name prefixes (see
    ``derive_zones``) and injected by the node; the file just holds the
    QR-payload → truck mapping. A ``zones`` key is tolerated but optional.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "qr_aliases" not in data:
        raise ValueError(f"{yaml_path}: missing required top-level 'qr_aliases'")
    data.setdefault("zones", {})
    return data


def _all_waypoints(zones_data: dict) -> set[str]:
    out: set[str] = set()
    for names in zones_data.get("zones", {}).values():
        out.update(names)
    return out


def _waypoint_class(zones_data: dict, name: str) -> Optional[str]:
    for cls, names in zones_data.get("zones", {}).items():
        if name in names:
            return cls
    return None


def resolve_qr_to_waypoint(zones_data: dict, qr_payload: str) -> Optional[str]:
    """Look up the truck waypoint encoded by a QR payload. Case-insensitive."""
    if not qr_payload:
        return None
    key = qr_payload.strip().upper()
    return zones_data.get("qr_aliases", {}).get(key)


def parse_mission(json_str: str, zones_data: dict) -> Optional[dict]:
    """Parse a /mission JSON string and resolve it into an execution-ready dict.

    On success returns:
        {
          "id":               <str>,
          "type":             <str>,         # one of MISSION_TYPES
          "candidate_queue":  <deque[str]>,  # waypoints to visit in order
          "scan_qr":          <bool>,        # whether SCAN_QR should run
          "destination":      <str | None>,  # None means "resolved from QR"
          "pickup_level":     <int>,         # 0..7
          "place_level":      <int>,         # 0..7
          "skip_alignment":   <bool>,
        }

    Returns None on any validation error. The state machine treats None as
    "publish MISSION_FAILED with reason 'invalid mission'".
    """
    if not json_str or not json_str.strip():
        return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    mtype = data.get("type")
    if mtype not in MISSION_TYPES:
        return None
    mid = str(data.get("id", "")) or "anonymous"

    all_wps = _all_waypoints(zones_data)
    zones = zones_data["zones"]

    if mtype == "ROLLER_TO_TRUCK":
        return _build_search_mission(
            data, zones, all_wps, mid,
            zone_category="rollers",
            default_pickup=DEFAULT_PICKUP_LEVELS["rollers"],
        )

    if mtype == "RACK_TO_TRUCK":
        return _build_search_mission(
            data, zones, all_wps, mid,
            zone_category="racks",
            default_pickup=DEFAULT_PICKUP_LEVELS["racks"],
        )

    if mtype == "SEARCH_ROLLERS":
        return _build_search_mission(
            data, zones, all_wps, mid,
            zone_category="rollers",
            default_pickup=DEFAULT_PICKUP_LEVELS["rollers"],
            search_only=True,
        )

    if mtype == "SEARCH_RACKS":
        return _build_search_mission(
            data, zones, all_wps, mid,
            zone_category="racks",
            default_pickup=DEFAULT_PICKUP_LEVELS["racks"],
            search_only=True,
        )

    if mtype == "PICK_ONLY":
        return _build_pick_only_mission(data, mid)

    # CUSTOM
    return _build_custom_mission(data, all_wps, mid)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_search_mission(
    data: dict,
    zones: dict,
    all_wps: set[str],
    mid: str,
    *,
    zone_category: str,
    default_pickup: int,
    search_only: bool = False,
) -> Optional[dict]:
    """Common builder for ROLLER_TO_TRUCK / RACK_TO_TRUCK.

    With ``search_only=True`` the mission stops once SEARCH locates a pallet
    (SEARCH returns the ``done`` outcome → MISSION_DONE) instead of continuing
    to PICK / NAV_TO_TRUCK / PLACE. Used by the SEARCH_ROLLERS / SEARCH_RACKS
    voice commands.
    """
    src = data.get("source", {})
    if not isinstance(src, dict):
        return None

    ordered = list(zones.get(zone_category, []))
    pool = set(ordered)
    if not pool:
        return None

    candidates_arg = src.get("candidates")
    if candidates_arg is None:
        # Visit every candidate in the class, in name-sorted order.
        queue = deque(ordered)
    else:
        if not isinstance(candidates_arg, list) or not candidates_arg:
            return None
        # Every named candidate must exist in the relevant pool.
        for w in candidates_arg:
            if not isinstance(w, str) or w not in pool:
                return None
        queue = deque(candidates_arg)

    pickup_level = _coerce_level(data.get("pickup_level"), default_pickup)
    if pickup_level is None:
        return None
    place_level = _coerce_level(data.get("place_level"), DEFAULT_PLACE_LEVEL_TRUCK)
    if place_level is None:
        return None

    return {
        "id":              mid,
        "type":            data["type"],
        "candidate_queue": queue,
        "scan_qr":         True,
        "destination":     None,            # resolved from QR
        "pickup_level":    pickup_level,
        "place_level":     place_level,
        "skip_alignment":  bool(data.get("skip_alignment", False)),
        "search_only":     search_only,
    }


def _build_pick_only_mission(data: dict, mid: str) -> Optional[dict]:
    """PICK_ONLY: skip SEARCH/nav and pick the pallet right in front of the robot.

    Used to test the PICK step in isolation — position the robot at a pallet,
    then it aligns, approaches to the LiDAR target, and lifts. Ends after the
    pick (no nav-to-truck / place). ``pick_only`` makes SEARCH short-circuit to
    PICK and PICK finish to MISSION_DONE.
    """
    pickup_level = _coerce_level(data.get("pickup_level"), DEFAULT_PICKUP_LEVELS["rollers"])
    if pickup_level is None:
        return None
    return {
        "id":              mid,
        "type":            "PICK_ONLY",
        "candidate_queue": deque(),       # unused (SEARCH is skipped)
        "scan_qr":         False,
        "destination":     None,
        "pickup_level":    pickup_level,
        "place_level":     DEFAULT_PLACE_LEVEL_TRUCK,
        "skip_alignment":  bool(data.get("skip_alignment", False)),
        "pick_only":       True,
    }


def _build_custom_mission(data: dict, all_wps: set[str], mid: str) -> Optional[dict]:
    src = data.get("source", {})
    if not isinstance(src, dict):
        return None
    src_wp = src.get("waypoint")
    if not isinstance(src_wp, str) or src_wp not in all_wps:
        return None

    dest = data.get("destination")
    dest_wp: Optional[str]
    if isinstance(dest, dict):
        dest_wp = dest.get("waypoint")
        if not isinstance(dest_wp, str) or dest_wp not in all_wps:
            return None
    elif isinstance(dest, str):
        # Allow "auto_from_qr" sentinel even in CUSTOM if scan_qr is enabled.
        if dest == "auto_from_qr":
            dest_wp = None
        elif dest in all_wps:
            dest_wp = dest
        else:
            return None
    else:
        return None

    scan_qr = bool(src.get("scan_qr", False))
    if dest_wp is None and not scan_qr:
        # No destination AND no QR scan ⇒ no way to know where to go.
        return None

    pickup_level = _coerce_level(data.get("pickup_level"), 1)
    if pickup_level is None:
        return None
    place_level = _coerce_level(data.get("place_level"), DEFAULT_PLACE_LEVEL_TRUCK)
    if place_level is None:
        return None

    return {
        "id":              mid,
        "type":            "CUSTOM",
        "candidate_queue": deque([src_wp]),
        "scan_qr":         scan_qr,
        "destination":     dest_wp,
        "pickup_level":    pickup_level,
        "place_level":     place_level,
        "skip_alignment":  bool(data.get("skip_alignment", False)),
    }


def _coerce_level(value, default: int) -> Optional[int]:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        lvl = value
    elif isinstance(value, float) and value.is_integer():
        lvl = int(value)
    else:
        return None
    if not (0 <= lvl <= 7):
        return None
    return lvl
