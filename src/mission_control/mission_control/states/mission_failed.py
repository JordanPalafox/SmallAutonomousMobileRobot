"""MISSION_FAILED — terminal failure state (with safe-stop)."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from yasmin import Blackboard

from mission_control.debug_wrapper import DebuggableState, DebugContext
from mission_control.bb_helpers import bb_get

logger = logging.getLogger(__name__)


class MissionFailed(DebuggableState):
    """Log the failure reason, leave the robot safe, then return to IDLE.

    Safe-stop: cancels navigation (publishes 'stop' on /goal_waypoint) and stops
    any docking maneuver (/alignment_start False) so a mid-motion/mid-dock
    failure doesn't leave the robot driving. The lifter is intentionally left
    where it is so a carried pallet isn't dropped on a failure.
    """

    def __init__(
        self,
        debug_ctx: DebugContext,
        publish_goal_fn: Optional[Callable[[str], None]] = None,
        publish_alignment_start_fn: Optional[Callable[[bool], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__("MISSION_FAILED", ["ok"], debug_ctx, abort_outcome="ok",
                         clears_abort=True, **kwargs)
        self._publish_goal = publish_goal_fn
        self._publish_align = publish_alignment_start_fn

    def run(self, blackboard: Blackboard) -> str:
        mission = bb_get(blackboard, "current_mission") or {}
        reason = bb_get(blackboard, "mission_error_reason") or "unspecified"
        logger.error(
            "[MISSION_FAILED] Mission %s aborted: %s",
            mission.get("id"), reason,
        )
        # Safe-stop.
        if self._publish_goal is not None:
            self._publish_goal("stop")
        if self._publish_align is not None:
            self._publish_align(False)
        return "ok"
