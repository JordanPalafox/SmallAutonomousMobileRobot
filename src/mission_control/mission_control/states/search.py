"""SEARCH — mission step 1: find the pallet.

Visits the mission's candidate waypoints in order; at each one it navigates
there and then watches ``/qr_detected`` for a freshly decoded payload. The
first candidate that yields a QR wins — its name and payload are left on the
blackboard for the pick / nav-to-truck steps.

This single state replaces the old PLAN_MISSION + NAV_TO_CANDIDATE + SCAN_QR +
NEXT_CANDIDATE search loop: the candidate loop is kept *internal* so the state
machine stays flat (one state per mission step). The QR decoder
(perception/qr_quad_alignment) publishes ``/qr_detected`` on every frame it
decodes, independent of ``/alignment_start``, so SEARCH can read the QR while
merely parked at a candidate — no docking motion is triggered here.
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
        found     — a candidate was reached and a fresh QR payload was read
                    (or scan_qr is disabled and the candidate was reached).
        done      — same success condition, but for a search_only mission
                    (SEARCH_ROLLERS / SEARCH_RACKS): stop here, no pick/deliver.
        not_found — invalid mission, or every candidate exhausted with no QR.
        stop      — abort raised.
    """

    def __init__(
        self,
        debug_ctx: DebugContext,
        publish_goal_fn: Callable[[str], None],
        scan_qr_timeout: float,
        **kwargs,
    ) -> None:
        super().__init__(
            "SEARCH", ["found", "not_found", "stop", "done"], debug_ctx,
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
        # go straight to PICK.
        if mission.get("pick_only"):
            blackboard["current_candidate"] = None
            blackboard["qr_value"] = None
            blackboard["resolved_dest"] = None
            logger.info("[SEARCH] pick_only — skipping search, going straight to PICK.")
            return "found"

        src_queue = mission.get("candidate_queue")
        if not src_queue:
            blackboard["mission_error_reason"] = "Candidate queue is empty."
            return "not_found"

        # Copy so the loop's popleft() doesn't mutate the parsed mission, and
        # so the dashboard snapshot sees the queue shrink as we search.
        queue: deque = deque(src_queue)
        blackboard["candidate_queue"] = queue
        blackboard["current_candidate"] = None
        blackboard["qr_value"] = None
        blackboard["resolved_dest"] = mission.get("destination")
        scan_qr = bool(mission.get("scan_qr", True))
        # SEARCH_ROLLERS / SEARCH_RACKS stop at the pallet instead of picking it.
        success = "done" if mission.get("search_only") else "found"

        logger.info(
            "[SEARCH] Mission %s — searching %d candidate(s): %s",
            mission.get("id"), len(queue), list(queue),
        )

        # Visit candidates in order until one yields a QR.
        while queue:
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

            if self._scan_qr_at(blackboard, candidate):
                return success
            if self._debug.aborted:
                self._publish_goal("stop")
                return "stop"
            logger.info("[SEARCH] no QR at %s — trying next candidate.", candidate)

        logger.warning("[SEARCH] All candidates exhausted, no pallet QR found.")
        blackboard["mission_error_reason"] = "No QR pallet found in any candidate location."
        return "not_found"

    # ------------------------------------------------------------------
    def _scan_qr_at(self, blackboard: Blackboard, candidate: str) -> bool:
        """Watch /qr_detected for a fresh payload at this candidate.

        Only payloads timestamped after the scan started count, so a QR seen
        while driving here doesn't satisfy the scan. Returns True on success;
        on abort it returns False and the caller re-checks the abort flag.
        """
        scan_started_at = time.monotonic()
        deadline = scan_started_at + self._scan_timeout
        logger.info("[SEARCH] scanning QR at %s (timeout=%.1fs)", candidate, self._scan_timeout)

        while True:
            if self._debug.aborted:
                return False
            self._debug.wait_if_paused()

            qr = bb_get(blackboard, "qr_detected")
            qr_t = bb_get(blackboard, "qr_detected_at", 0.0)
            if qr and qr_t >= scan_started_at:
                logger.info("[SEARCH] found QR %r at %s", qr, candidate)
                blackboard["qr_value"] = qr
                return True

            if time.monotonic() > deadline:
                return False

            time.sleep(_POLL_INTERVAL)
