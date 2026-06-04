"""PLACE — mission step 4: lower the lifter to release the pallet."""

from __future__ import annotations

import logging
from typing import Callable

from yasmin import Blackboard

from mission_control.debug_wrapper import DebuggableState, DebugContext
from mission_control.bb_helpers import bb_get
from mission_control.states._actions import drive_lifter

logger = logging.getLogger(__name__)


class Place(DebuggableState):
    """Step 4 — *deja*: lower the lifter to ``place_level`` to drop the pallet.

    Outcomes:
        placed — lifter reached place_level.
        failed — lifter timeout.
        stop   — abort raised.
    """

    def __init__(
        self,
        debug_ctx: DebugContext,
        publish_lifter_fn: Callable[[int], None],
        lifter_timeout: float,
        **kwargs,
    ) -> None:
        super().__init__(
            "PLACE", ["placed", "failed", "stop"], debug_ctx,
            abort_outcome="stop", **kwargs,
        )
        self._publish_lifter = publish_lifter_fn
        self._lifter_timeout = float(lifter_timeout)

    def run(self, blackboard: Blackboard) -> str:
        mission = bb_get(blackboard, "current_mission") or {}
        level = int(mission.get("place_level", 0))
        outcome = drive_lifter(
            self._debug, blackboard, self._publish_lifter,
            level, self._lifter_timeout, tag="PLACE",
        )
        if outcome == "stop":
            return "stop"
        if outcome == "timeout":
            return "failed"
        return "placed"
