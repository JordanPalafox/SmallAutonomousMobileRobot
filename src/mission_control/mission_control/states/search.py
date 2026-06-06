"""SEARCH — mission step 1: find the pallet.

Visits the mission's candidate waypoints in order; at each one it navigates
there and ONLY THEN — parked at the waypoint, facing the pallet — opens the QR
window, watching ``/qr_detected`` for a payload decoded from that pose. Anything
decoded DURING the approach is ignored on purpose: with the rollers in a row a
neighbour's QR drifts through the frame en route, and accepting it would hand
off to PICK at the wrong candidate before the robot ever arrives. The first
candidate that yields a post-arrival QR wins — its name and payload are left on
the blackboard for the pick / nav-to-truck steps, and SEARCH hands off to PICK
with the robot still facing the pallet, so PICK's alignment only has to
fine-tune (not recover from a big offset).

No oscillation here on purpose: sweeping to hunt for the QR made the robot
decode it far off-centre (in the lateral edge of the frame) and enter PICK badly
aligned, triggering an unnecessary recovery rotation in docking. Instead SEARCH
just waits. If a roller doesn't yield a QR within ``scan_qr_timeout`` it moves
to the next one; once every roller is exhausted it refills the queue and keeps
searching — SEARCH never leaves on "no QR", only on a decoded QR (→ PICK), an
abort, or an invalid/empty mission.

The QR decoder (perception/qr_quad_alignment) publishes ``/qr_detected`` on
every frame it decodes, independent of ``/alignment_start``, so SEARCH reads the
QR while merely parked at a candidate — no motion is commanded here.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable

from yasmin import Blackboard

from mission_control.debug_wrapper import DebuggableState, DebugContext
from mission_control.bb_helpers import bb_get
from mission_control.states._actions import navigate

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.1


class Search(DebuggableState):
    """Mission step 1 — *buscar el pallet*.

    Outcomes:
        found      — a candidate was reached and a fresh QR payload was read
                     (or scan_qr is disabled and the candidate was reached).
                     Roller / custom flow → PICK.
        found_rack — same success condition, but the mission is RACK_TO_TRUCK
                     (mission 2) → PICK_FROM_RACK (rack lifter levels + rack QR
                     dock calibration).
        done       — same success condition, but for a search_only mission
                     (SEARCH_ROLLERS / SEARCH_RACKS): stop here, no pick/deliver.
        not_found  — invalid mission, or empty candidate queue.
        stop       — abort raised.
    """

    def __init__(
        self,
        debug_ctx: DebugContext,
        publish_goal_fn: Callable[[str], None],
        scan_qr_timeout: float,
        **kwargs,
    ) -> None:
        super().__init__(
            "SEARCH", ["found", "found_rack", "not_found", "stop", "done"], debug_ctx,
            abort_outcome="stop", **kwargs,
        )
        self._publish_goal = publish_goal_fn
        self._scan_timeout = float(scan_qr_timeout)

    # ------------------------------------------------------------------
    def run(self, blackboard: Blackboard) -> str:
        # Seed execution state from the parsed mission (absorbs PLAN_MISSION).
        mission = bb_get(blackboard, "current_mission")
        if mission is None:
            blackboard["mission_error_reason"] = "JSON failed to parse or violated schema."
            return "not_found"

        # PICK_ONLY test: robot is already at the pallet — skip search/nav,
        # go straight to PICK (roller) or PICK_FROM_RACK (rack).
        if mission.get("pick_only"):
            blackboard["current_candidate"] = None
            blackboard["qr_value"] = None
            blackboard["resolved_dest"] = None
            if mission.get("pick_rack"):
                logger.info("[SEARCH] pick_only (rack) — skipping search, going straight to PICK_FROM_RACK.")
                return "found_rack"
            logger.info("[SEARCH] pick_only — skipping search, going straight to PICK.")
            return "found"

        # RELEASE_ONLY test: pretend a QR was already decoded; skip search (and
        # PICK skips too) so the SM jumps to the truck-zone delivery (RELEASE_LOAD).
        if mission.get("release_only"):
            blackboard["current_candidate"] = None
            blackboard["qr_value"] = mission.get("qr") or "Popsi"
            blackboard["resolved_dest"] = None
            logger.info("[SEARCH] release_only — skipping search; pretend QR=%r.",
                        blackboard["qr_value"])
            return "found"

        src_queue = mission.get("candidate_queue")
        if not src_queue:
            blackboard["mission_error_reason"] = "Candidate queue is empty."
            return "not_found"

        blackboard["current_candidate"] = None
        blackboard["qr_value"] = None
        blackboard["resolved_dest"] = mission.get("destination")
        scan_qr = bool(mission.get("scan_qr", True))
        # Success outcome routes the pallet-found handoff:
        #   SEARCH_ROLLERS / SEARCH_RACKS (search_only) → done (stop, no pick).
        #   RACK_TO_TRUCK (mission 2)                   → found_rack → PICK_FROM_RACK.
        #   everything else (roller / custom)           → found → PICK.
        if mission.get("search_only"):
            success = "done"
        elif mission.get("type") == "RACK_TO_TRUCK":
            success = "found_rack"
        else:
            success = "found"

        logger.info(
            "[SEARCH] Mission %s — searching candidate(s): %s",
            mission.get("id"), list(src_queue),
        )

        # Keep searching until a QR is decoded (→ PICK) or aborted. When the
        # queue is exhausted without a QR we REFILL it and re-scan, so a missing
        # pallet keeps the machine in SEARCH instead of failing to IDLE.
        queue: deque = deque(src_queue)
        blackboard["candidate_queue"] = queue
        while True:
            if not queue:
                logger.info("[SEARCH] no QR in any candidate yet — re-scanning the rollers.")
                queue = deque(src_queue)
                blackboard["candidate_queue"] = queue

            if self._debug.aborted:
                self._publish_goal("stop")
                return "stop"

            candidate = queue.popleft()
            blackboard["current_candidate"] = candidate

            nav_outcome = navigate(self._debug, blackboard, self._publish_goal, candidate, tag="SEARCH")
            if nav_outcome == "stop":
                return "stop"
            if nav_outcome == "error":
                logger.warning("[SEARCH] nav failed at %s — trying next candidate.", candidate)
                continue

            # Arrived. Look for a QR (unless this mission opts out of scanning).
            if not scan_qr:
                logger.info("[SEARCH] scan_qr=false — accepting %s without QR.", candidate)
                return success

            # The QR window opens ONLY now that we've reached the waypoint and are
            # parked facing the pallet. Anything decoded DURING the approach is
            # ignored: it is usually a neighbouring roller's QR drifting through
            # the frame, and accepting it would send us to PICK at the wrong
            # candidate before we ever arrive here. valid_from = arrival time, so
            # only a fresh decode read from THIS parked pose counts.
            arrived_at = time.monotonic()
            if self._scan_qr_at(blackboard, candidate, valid_from=arrived_at):
                return success
            if self._debug.aborted:
                self._publish_goal("stop")
                return "stop"
            # Per-candidate budget expired with no QR → next roller (or re-cycle).
            logger.info("[SEARCH] no QR at %s within %.1fs — trying next candidate.",
                        candidate, self._scan_timeout)

    # ------------------------------------------------------------------
    def _scan_qr_at(self, blackboard: Blackboard, candidate: str,
                    valid_from: float) -> bool:
        """Stay parked and watch /qr_detected for a payload decoded AFTER the
        robot reached THIS candidate (timestamp >= ``valid_from`` = arrival
        time), then return True on the first qualifying decode. No motion is
        commanded — the robot keeps facing the pallet so PICK starts roughly
        centred.

        ``valid_from`` is the ARRIVAL time, so a QR glimpsed during the approach
        (e.g. a neighbouring roller's pallet) or read at a PREVIOUS candidate
        (both older timestamps) does NOT count — only a decode produced from this
        parked pose. The per-candidate budget is ``scan_timeout``: >0 returns
        False on expiry so the caller moves to the next roller; <=0 waits
        indefinitely at this candidate (the search then never advances past it).

        Returns True on success; on abort it returns False and the caller
        re-checks the abort flag.
        """
        scan_started_at = time.monotonic()
        wait_forever = self._scan_timeout <= 0.0
        deadline = None if wait_forever else scan_started_at + self._scan_timeout

        logger.info("[SEARCH] parked at %s, waiting for QR (timeout=%s)",
                    candidate, "none" if wait_forever else f"{self._scan_timeout:.1f}s")
        while True:
            if self._debug.aborted:
                return False
            self._debug.wait_if_paused()

            qr = bb_get(blackboard, "qr_detected")
            qr_t = bb_get(blackboard, "qr_detected_at", 0.0)
            if qr and qr_t >= valid_from:
                logger.info("[SEARCH] decoded QR %r at %s", qr, candidate)
                blackboard["qr_value"] = qr
                return True

            if deadline is not None and time.monotonic() > deadline:
                return False

            time.sleep(_POLL_INTERVAL)
