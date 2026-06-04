"""MISSION_DONE — terminal success state."""

from __future__ import annotations

import logging

from yasmin import Blackboard

from mission_control.debug_wrapper import DebuggableState, DebugContext
from mission_control.bb_helpers import bb_get

logger = logging.getLogger(__name__)


class MissionDone(DebuggableState):
    """Publish a success record then return to IDLE."""

    def __init__(self, debug_ctx: DebugContext, **kwargs) -> None:
        super().__init__("MISSION_DONE", ["ok"], debug_ctx, abort_outcome="ok",
                         clears_abort=True, **kwargs)

    def run(self, blackboard: Blackboard) -> str:
        mission = bb_get(blackboard, "current_mission") or {}
        logger.info(
            "[MISSION_DONE] Mission %s completed → resolved_dest=%s",
            mission.get("id"), bb_get(blackboard, "resolved_dest"),
        )
        return "ok"
