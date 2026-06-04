"""IDLE — wait for /mission and clear residue from the previous run."""

from __future__ import annotations

import logging
import time
from typing import Callable

from yasmin import Blackboard

from mission_control.debug_wrapper import DebuggableState, DebugContext
from mission_control.bb_helpers import bb_get
from mission_control.mission_parser import parse_mission

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.1


class Idle(DebuggableState):
    """Wait for an incoming mission and hand it to SEARCH.

    Outcomes:
        mission_received — a valid JSON arrived; pre-parsed dict is on the
                           blackboard under ``current_mission`` (None on parse
                           error → still transitions, and SEARCH decides).
    """

    def __init__(self, debug_ctx: DebugContext, zones_data: dict, **kwargs) -> None:
        super().__init__("IDLE", ["mission_received"], debug_ctx, clears_abort=True, **kwargs)
        self._zones = zones_data

    def run(self, blackboard: Blackboard) -> str:
        # Clear residue from the previous mission so stale data doesn't leak.
        for key in (
            "current_mission",
            "candidate_queue",
            "current_candidate",
            "qr_value",
            "resolved_dest",
            "mission_error_reason",
        ):
            blackboard[key] = None
        blackboard["nav_status_prefix"] = "IDLE"

        logger.info("[IDLE] Waiting for /mission …")

        while True:
            if self._debug.aborted:
                # A stop pressed while already idle: just consume it and keep
                # waiting — do NOT advance to SEARCH.
                self._debug.set_abort(False)

            self._debug.wait_if_paused()

            raw = bb_get(blackboard, "mission_raw")
            if raw:
                parsed = parse_mission(raw, self._zones)
                # Always consume the raw payload so the same string isn't
                # re-parsed forever; SEARCH sees parse_failure via None.
                blackboard["mission_raw"] = None
                blackboard["current_mission"] = parsed
                if parsed is None:
                    logger.warning("[IDLE] Mission JSON rejected by parser.")
                else:
                    logger.info(
                        "[IDLE] Mission accepted: id=%s type=%s candidates=%s",
                        parsed["id"], parsed["type"],
                        list(parsed["candidate_queue"]),
                    )
                return "mission_received"

            time.sleep(_POLL_INTERVAL)
