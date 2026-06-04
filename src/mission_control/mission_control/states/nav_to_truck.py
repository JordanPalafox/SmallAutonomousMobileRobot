"""NAV_TO_TRUCK — mission step 3: resolve the truck from the QR, then drive there.

Merges the old RESOLVE_DESTINATION + NAV_TO_DESTINATION states: the QR→truck
lookup is done inline at the start of the navigation, matching the intent
"navigate to the truck depending on the QR that was read".
"""

from __future__ import annotations

import logging
from typing import Callable

from yasmin import Blackboard

from mission_control.debug_wrapper import DebuggableState, DebugContext
from mission_control.bb_helpers import bb_get
from mission_control.mission_parser import resolve_qr_to_waypoint
from mission_control.states._actions import navigate

logger = logging.getLogger(__name__)


class NavToTruck(DebuggableState):
    """Step 3 — *navega al camión según el QR*.

    Outcomes:
        arrived — nav_node reported ARRIVED at the truck.
        failed  — QR did not map to a truck, or navigation error.
        stop    — abort raised.
    """

    def __init__(
        self,
        debug_ctx: DebugContext,
        publish_goal_fn: Callable[[str], None],
        zones_data: dict,
        **kwargs,
    ) -> None:
        super().__init__(
            "NAV_TO_TRUCK", ["arrived", "failed", "stop"], debug_ctx,
            abort_outcome="stop", **kwargs,
        )
        self._publish_goal = publish_goal_fn
        self._zones = zones_data

    def run(self, blackboard: Blackboard) -> str:
        mission = bb_get(blackboard, "current_mission") or {}

        # A pre-set destination (CUSTOM) wins; otherwise resolve from the QR.
        dest = mission.get("destination")
        if not dest:
            qr = bb_get(blackboard, "qr_value")
            dest = resolve_qr_to_waypoint(self._zones, qr or "")
            if dest is None:
                logger.error("[NAV_TO_TRUCK] QR %r not in qr_aliases.", qr)
                blackboard["mission_error_reason"] = f"QR payload {qr!r} not in qr_aliases"
                return "failed"
            logger.info("[NAV_TO_TRUCK] QR %r → %s", qr, dest)
        blackboard["resolved_dest"] = dest

        outcome = navigate(self._debug, blackboard, self._publish_goal, dest, tag="NAV_TO_TRUCK")
        if outcome == "arrived":
            return "arrived"
        if outcome == "stop":
            return "stop"
        return "failed"   # nav 'error'
