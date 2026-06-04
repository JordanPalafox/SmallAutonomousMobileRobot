"""PICK — mission step 2: dock onto the pallet, then a timed pick maneuver.

Sequence:
    1. QR alignment (qr_quad_alignment docking) to centre on the pallet.
    2. Set the lifter to ``entry_level`` (fork height to slide under the pallet).
    3. Creep forward, stopping by VISION (Electric-80 logo at target distance,
       /approach_stop/should_stop) before contact — with wheel stall + a
       ``forward_time`` time limit as safety fallbacks.
    4. Set the lifter to ``lift_level`` to take the pallet's weight.
    5. Drive backward for ``reverse_time`` s to pull the pallet clear.
    6. Set the lifter to ``transport_level`` (carry height) — then done.
"""

from __future__ import annotations

import logging
from typing import Callable

from yasmin import Blackboard

from mission_control.debug_wrapper import DebuggableState, DebugContext
from mission_control.bb_helpers import bb_get
from mission_control.states._actions import (
    run_alignment, drive_for_time, drive_until_approach_stop, drive_lifter,
)

logger = logging.getLogger(__name__)


class Pick(DebuggableState):
    """Step 2 — *recoge*: align, drive in, lift, back out.

    Outcomes:
        picked — full maneuver done → continue mission.
        done   — same, but PICK_ONLY test → end mission.
        failed — alignment failed/timeout, or lifter timeout.
        stop   — abort raised.
    """

    def __init__(
        self,
        debug_ctx: DebugContext,
        publish_alignment_start_fn: Callable[[bool], None],
        publish_lifter_fn: Callable[[int], None],
        publish_cmd_fn: Callable[[float, float], None],
        alignment_timeout: float,
        lifter_timeout: float,
        drive_speed: float,
        forward_time: float,
        reverse_time: float,
        entry_level: int,
        lift_level: int,
        stall_grace: float,
        stall_speed: float,
        stall_ticks: int,
        vision_stop: bool,
        vision_fresh_s: float,
        transport_level: int,
        **kwargs,
    ) -> None:
        super().__init__(
            "PICK", ["picked", "done", "failed", "stop"], debug_ctx,
            abort_outcome="stop", **kwargs,
        )
        self._publish_align = publish_alignment_start_fn
        self._publish_lifter = publish_lifter_fn
        self._publish_cmd = publish_cmd_fn
        self._align_timeout = float(alignment_timeout)
        self._lifter_timeout = float(lifter_timeout)
        self._drive_speed = float(drive_speed)
        self._forward_time = float(forward_time)
        self._reverse_time = float(reverse_time)
        self._entry_level = int(entry_level)
        self._lift_level = int(lift_level)
        self._stall_grace = float(stall_grace)
        self._stall_speed = float(stall_speed)
        self._stall_ticks = int(stall_ticks)
        self._vision_stop = bool(vision_stop)
        self._vision_fresh = float(vision_fresh_s)
        self._transport_level = int(transport_level)

    def run(self, blackboard: Blackboard) -> str:
        mission = bb_get(blackboard, "current_mission") or {}

        # --- 1) align onto the pallet (QR docking) ---
        if mission.get("skip_alignment"):
            logger.info("[PICK] skip_alignment=true — skipping docking.")
        else:
            outcome = run_alignment(
                self._debug, blackboard, self._publish_align,
                self._align_timeout, tag="PICK",
            )
            if outcome == "stop":
                return "stop"
            if outcome == "failed":
                return "failed"

        # --- 2) raise forks to the entry height (slide under the pallet) ---
        outcome = drive_lifter(
            self._debug, blackboard, self._publish_lifter,
            self._entry_level, self._lifter_timeout, tag="PICK entry",
        )
        if outcome == "stop":
            return "stop"
        if outcome == "timeout":
            return "failed"

        # --- 3) creep forward toward the pallet: stop by VISION (Electric-80
        #         logo at target distance) BEFORE touching the load, so the
        #         motor never stalls into it and browns out the Jetson. Wheel
        #         stall and the time limit remain as safety fallbacks. ---
        if drive_until_approach_stop(
                self._debug, blackboard, self._publish_cmd,
                self._drive_speed, 0.0, self._forward_time,
                grace=self._stall_grace, stall_speed=self._stall_speed,
                stall_ticks=self._stall_ticks,
                vision_enabled=self._vision_stop,
                vision_fresh_s=self._vision_fresh,
                tag="PICK fwd") == "stop":
            return "stop"

        # --- 4) lift the pallet (change to lift level) ---
        outcome = drive_lifter(
            self._debug, blackboard, self._publish_lifter,
            self._lift_level, self._lifter_timeout, tag="PICK lift",
        )
        if outcome == "stop":
            return "stop"
        if outcome == "timeout":
            return "failed"

        # --- 5) back out with the pallet (timed reverse) ---
        if drive_for_time(self._debug, blackboard, self._publish_cmd,
                          -self._drive_speed, 0.0, self._reverse_time,
                          tag="PICK rev") == "stop":
            return "stop"

        # --- 6) raise the lifter to the transport level before finishing ---
        outcome = drive_lifter(
            self._debug, blackboard, self._publish_lifter,
            self._transport_level, self._lifter_timeout, tag="PICK transport",
        )
        if outcome == "stop":
            return "stop"
        if outcome == "timeout":
            return "failed"

        return "done" if mission.get("pick_only") else "picked"
