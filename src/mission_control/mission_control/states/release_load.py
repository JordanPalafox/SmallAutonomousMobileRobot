"""RELEASE_LOAD — mission step 5: deliver the pallet to the truck whose logo
matches the pallet's QR, then release it.

Runs after NAV_TO_TRUCK has parked the robot at the truck-zone vantage point
(``truck_1``). From there:

    1. The camera's field of view may not hold all three logos at once, so the
       robot OSCILLATES in place (turns a little right, a little left, widening
       the swing) while watching ``/logo_order`` (from perception/logo_classifier,
       a YOLO model). Each frame is the detector's left→right order of whatever
       logos are currently in view (often only a subset). We ACCUMULATE the
       pairwise "X is left of Y" votes across the whole sweep; once they chain
       into a strict total order of all logos {amazon, pepsi, walmart} we have
       the order — even if no single frame ever showed all three together.
    2. That left→right order maps positionally to the truck waypoints
       (order[0]→truck_2, order[1]→truck_3, order[2]→truck_4).
    3. The QR read at the roller (``qr_value``, e.g. "Popsi") is normalised to a
       logo class (pepsi) via ``logo_aliases`` and matched to the truck holding
       that logo.
    4. The robot navigates to that truck and lowers the lifter to release_level
       to drop the pallet.

A CUSTOM mission with an explicit destination skips the logo matching (it was
already driven to its destination by NAV_TO_TRUCK) and just releases.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List

from yasmin import Blackboard

from mission_control.debug_wrapper import DebuggableState, DebugContext
from mission_control.bb_helpers import bb_get
from mission_control.mission_parser import resolve_logo_alias
from mission_control.states._actions import navigate, drive_lifter, oscillate_until

logger = logging.getLogger(__name__)

# Ignore a /logo_order older than this (s): a frozen value from a dead detector
# must not accumulate "stability" and trigger a false acquisition.
_LOGO_FRESH_S = 1.5


class ReleaseLoad(DebuggableState):
    """Step 5 — *suelta la carga en el camión del logo correcto*.

    Outcomes:
        released — delivered to the matched truck and lifter lowered.
        failed   — logos not seen in time, QR not aliased, or nav/lifter error.
        stop     — abort raised.
    """

    def __init__(
        self,
        debug_ctx: DebugContext,
        publish_goal_fn: Callable[[str], None],
        publish_cmd_fn: Callable[[float, float], None],
        publish_lifter_fn: Callable[[int], None],
        zones_data: dict,
        logo_truck_waypoints: List[str],
        osc_angular_speed: float,
        osc_sweep_start_deg: float,
        osc_sweep_step_deg: float,
        osc_sweep_max_deg: float,
        osc_timeout: float,
        logo_stable_frames: int,
        release_level: int,
        lifter_timeout: float,
        **kwargs,
    ) -> None:
        super().__init__(
            "RELEASE_LOAD", ["released", "failed", "stop"], debug_ctx,
            abort_outcome="stop", **kwargs,
        )
        self._publish_goal = publish_goal_fn
        self._publish_cmd = publish_cmd_fn
        self._publish_lifter = publish_lifter_fn
        self._zones = zones_data
        self._logo_truck_wps = list(logo_truck_waypoints)
        self._osc_speed = float(osc_angular_speed)
        self._sweep_start = float(osc_sweep_start_deg)
        self._sweep_step = float(osc_sweep_step_deg)
        self._sweep_max = float(osc_sweep_max_deg)
        self._osc_timeout = float(osc_timeout)
        self._stable_frames = max(1, int(logo_stable_frames))
        self._release_level = int(release_level)
        self._lifter_timeout = float(lifter_timeout)
        # The logo classes the detector knows = the distinct alias targets.
        self._expected_logos = {
            str(v).strip().lower()
            for v in (zones_data.get("logo_aliases") or {}).values()
        } or {"amazon", "pepsi", "walmart"}
        # Per-run scan state (reset at the start of run()).
        # _pair_counts[(x, y)] = how many FRESH frames showed logo x left of y.
        # _seen_logos = logos seen at least once in a usable frame.
        self._pair_counts: dict[tuple[str, str], int] = {}
        self._seen_logos: set[str] = set()
        self._last_at = 0.0
        self._captured: List[str] | None = None
        self._last_scan_log = 0.0   # throttle the per-poll scan diagnostics
        self._last_info_size = 0    # watchdog: distinct logos+pairs seen so far

    # ------------------------------------------------------------------
    def run(self, blackboard: Blackboard) -> str:
        mission = bb_get(blackboard, "current_mission") or {}

        # CUSTOM: an explicit destination was already reached by NAV_TO_TRUCK;
        # there's no 3-logo scene to match, so just release here.
        if mission.get("destination"):
            logger.info("[RELEASE_LOAD] explicit destination — releasing in place.")
            return self._release(blackboard)

        # --- match the QR's company to a logo class ---
        qr = bb_get(blackboard, "qr_value")
        target_logo = resolve_logo_alias(self._zones, qr or "")
        if target_logo is None:
            blackboard["mission_error_reason"] = f"QR {qr!r} not in logo_aliases"
            logger.error("[RELEASE_LOAD] %s", blackboard["mission_error_reason"])
            return "failed"
        logger.info("[RELEASE_LOAD] QR %r → logo %r; scanning for the %d truck logos %s …",
                    qr, target_logo, len(self._logo_truck_wps),
                    sorted(self._expected_logos))

        # --- resolve the trucks' left→right logo order by oscillating ---
        # First make nav_node yield /cmd_vel_in: right after arriving it lingers
        # ~1 s in ARRIVED still publishing a zero Twist, which would fight our
        # rotation. Cancelling the goal drops it to IDLE (where it publishes
        # nothing), so the oscillation owns /cmd_vel_in cleanly.
        self._publish_goal("stop")
        self._pair_counts = {}
        self._seen_logos = set()
        self._last_at = 0.0
        self._captured = None
        self._last_scan_log = 0.0
        self._last_info_size = 0
        outcome = oscillate_until(
            self._debug, blackboard, self._publish_cmd, self._logo_order_resolved,
            angular_speed=self._osc_speed, sweep_start_deg=self._sweep_start,
            sweep_step_deg=self._sweep_step, sweep_max_deg=self._sweep_max,
            timeout=self._osc_timeout, tag="RELEASE_LOAD scan",
            progress_fn=self._scan_progressing,
        )
        if outcome == "stop":
            return "stop"
        if outcome == "timeout":
            blackboard["mission_error_reason"] = (
                f"could not resolve the L→R order of {sorted(self._expected_logos)} "
                f"(seen: {sorted(self._seen_logos)}, votes: {self._fmt_pairs()})"
            )
            logger.error("[RELEASE_LOAD] %s", blackboard["mission_error_reason"])
            return "failed"
        if outcome != "found":   # defensive: oscillate_until only yields found/timeout/stop
            blackboard["mission_error_reason"] = f"unexpected scan outcome {outcome!r}"
            return "failed"

        order = self._captured or []
        # --- map left→right logos to the truck waypoints, pick the match ---
        truck_by_logo = dict(zip(order, self._logo_truck_wps))
        dest = truck_by_logo.get(target_logo)
        if dest is None:
            blackboard["mission_error_reason"] = (
                f"matched logo {target_logo!r} not among detected {order}"
            )
            logger.error("[RELEASE_LOAD] %s", blackboard["mission_error_reason"])
            return "failed"
        blackboard["resolved_dest"] = dest
        logger.info("[RELEASE_LOAD] logos L→R %s → %s | QR logo %s → %s",
                    order, self._logo_truck_wps, target_logo, dest)

        # --- drive to the matched truck ---
        nav = navigate(self._debug, blackboard, self._publish_goal, dest, tag="RELEASE_LOAD")
        if nav == "stop":
            return "stop"
        if nav == "error":
            return "failed"

        # --- release the pallet ---
        return self._release(blackboard)

    # ------------------------------------------------------------------
    def _release(self, blackboard: Blackboard) -> str:
        outcome = drive_lifter(
            self._debug, blackboard, self._publish_lifter,
            self._release_level, self._lifter_timeout, tag="RELEASE_LOAD release",
        )
        # An abort mid-drop leaves the lifter at whatever level it reached (no
        # command is forced) — consistent with MISSION_FAILED never commanding a
        # drop on failure. Re-running the mission re-issues release_level.
        if outcome == "stop":
            return "stop"
        if outcome == "timeout":
            return "failed"
        return "released"

    # ------------------------------------------------------------------
    def _scan_log(self, msg: str) -> None:
        """Throttled (~1 Hz) diagnostic for the logo scan, so a live test shows
        WHY the predicate is/isn't passing without spamming at the 10 Hz poll."""
        now = time.monotonic()
        if now - self._last_scan_log >= 1.0:
            self._last_scan_log = now
            logger.info("[RELEASE_LOAD] scan: %s", msg)

    def _scan_progressing(self, blackboard: Blackboard) -> bool:
        """Watchdog signal for the oscillation: True while the sweep is still
        DISCOVERING new information (a new logo, or a new left-of pair). It
        saturates once every visible relationship has been observed, so the scan
        only times out when it has stopped learning for ``osc_timeout`` seconds —
        a slow reveal (sweeping to bring the 3rd truck into view) keeps it alive,
        which a hard wall-clock cap would have cut off (→ MISSION_FAILED → IDLE)."""
        size = len(self._seen_logos) + len(self._pair_counts)
        if size > self._last_info_size:
            self._last_info_size = size
            return True
        return False

    def _fmt_pairs(self) -> str:
        """Compact view of the accumulated 'x left of y' votes, for diagnostics."""
        if not self._pair_counts:
            return "(none)"
        return ", ".join(
            f"{x}<{y}:{c}" for (x, y), c in sorted(self._pair_counts.items())
        )

    def _accumulate_frame(self, blackboard: Blackboard) -> None:
        """Fold the latest FRESH /logo_order frame into the pairwise tally.

        Each frame is the detector's left→right order of the logos currently in
        view (often a subset). Within one frame the cx ordering is reliable, so
        EVERY earlier→later pair is a 'left of' vote. Only NEW messages count
        (timestamp advanced) and only fresh ones (a frozen value from a dead
        detector is ignored). Frames with a duplicated logo are dropped as noise.
        """
        order = bb_get(blackboard, "logo_order")
        order_at = bb_get(blackboard, "logo_order_at", 0.0)

        if not isinstance(order, list):
            self._scan_log("no /logo_order received yet (is logo_classifier running?)")
            return
        if (time.monotonic() - order_at) > _LOGO_FRESH_S:
            self._scan_log(f"stale /logo_order: {order}")
            return
        if order_at == self._last_at:
            return                       # not a new message — don't double-count
        self._last_at = order_at

        # Keep only known logos, preserving the detector's left→right order.
        seq = [n for n in (str(x).strip().lower() for x in order)
               if n in self._expected_logos]
        if not seq:
            return
        if len(set(seq)) != len(seq):
            self._scan_log(f"duplicate logo in frame, dropping: {seq}")
            return

        for n in seq:
            self._seen_logos.add(n)
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                key = (seq[i], seq[j])
                self._pair_counts[key] = self._pair_counts.get(key, 0) + 1

    def _resolve_order(self) -> "List[str] | None":
        """Derive a strict total left→right order of ALL expected logos from the
        accumulated votes, or None if still ambiguous.

        An edge x→y ('x left of y') is trusted once it has ``logo_stable_frames``
        confirming frames AND outvotes the opposite direction. A unique
        topological sort over those edges is the order — transitivity ties in a
        logo the camera never co-framed with an extreme (amazon<pepsi & pepsi<
        walmart ⇒ amazon<walmart, never needing amazon and walmart in one frame).
        """
        if self._seen_logos != self._expected_logos:
            return None                  # haven't seen every logo even once yet

        support = self._stable_frames
        nodes = set(self._expected_logos)
        edges: set[tuple[str, str]] = set()
        for x in nodes:
            for y in nodes:
                if x == y:
                    continue
                fwd = self._pair_counts.get((x, y), 0)
                rev = self._pair_counts.get((y, x), 0)
                if fwd >= support and fwd > rev:
                    edges.add((x, y))

        # Kahn topological sort; require EXACTLY one in-degree-0 node each step,
        # else the order is ambiguous (or contradictory) and we wait for more.
        indeg = {n: 0 for n in nodes}
        for (_, y) in edges:
            indeg[y] += 1
        remaining = set(nodes)
        order: List[str] = []
        while remaining:
            zero = [n for n in remaining if indeg[n] == 0]
            if len(zero) != 1:
                return None
            n = zero[0]
            order.append(n)
            remaining.discard(n)
            for (a, b) in edges:
                if a == n and b in remaining:
                    indeg[b] -= 1
        return order

    def _logo_order_resolved(self, blackboard: Blackboard) -> bool:
        """oscillate_until predicate: True once the full L→R logo order is known.

        Sets ``_captured`` to that order so run() can map it to the trucks.
        """
        self._accumulate_frame(blackboard)
        order = self._resolve_order()
        if order is None:
            self._scan_log(
                f"resolving order: seen={sorted(self._seen_logos)} "
                f"votes={self._fmt_pairs()}"
            )
            return False
        self._captured = order
        logger.info("[RELEASE_LOAD] L→R order resolved: %s (votes: %s)",
                    order, self._fmt_pairs())
        return True
