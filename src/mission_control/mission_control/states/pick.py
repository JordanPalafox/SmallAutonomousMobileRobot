"""PICK — mission step 2: dock onto the pallet, then a timed pick maneuver.

Sequence:
    1. Set the lifter to ``entry_level`` (fork height to slide under the
       pallet) BEFORE docking, so the forks are staged before we centre.
    2. QR alignment (qr_quad_alignment docking) to centre on the pallet.
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
        reverse_speed: float,
        entry_level: int,
        lift_level: int,
        stall_grace: float,
        stall_speed: float,
        stall_ticks: int,
        vision_stop: bool,
        vision_fresh_s: float,
        transport_level: int,
        state_name: str = "PICK",
        center_kp: float = 0.0,
        center_w_max: float = 0.10,
        center_deadband_px: float = 12.0,
        center_fresh_s: float = 1.0,
        vision_stale_slow_s: float = 0.4,
        vision_stale_factor: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(
            state_name, ["picked", "done", "failed", "stop"], debug_ctx,
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
        self._reverse_speed = float(reverse_speed)
        self._entry_level = int(entry_level)
        self._lift_level = int(lift_level)
        self._stall_grace = float(stall_grace)
        self._stall_speed = float(stall_speed)
        self._stall_ticks = int(stall_ticks)
        self._vision_stop = bool(vision_stop)
        self._vision_fresh = float(vision_fresh_s)
        self._transport_level = int(transport_level)
        # Logo CENTERING during the approach creep (center_kp>0): steer on the
        # logo's lateral error so the lifter enters the pallet straight. Used by
        # both the roller (this base) and the rack (PickFromRack) — each gets its
        # own gain via params; 0 = straight creep (no centering).
        self._center_kp = float(center_kp)
        self._center_w_max = float(center_w_max)
        self._center_deadband_px = float(center_deadband_px)
        self._center_fresh_s = float(center_fresh_s)
        # Camera-freeze safety for the approach creep: if the vision feed stops
        # updating for vision_stale_slow_s, scale the forward speed by
        # vision_stale_factor (<1 slows, 0 holds) so a frozen frame can't carry
        # the robot past the ideal stop. factor=1.0 disables it.
        self._vision_stale_slow_s = float(vision_stale_slow_s)
        self._vision_stale_factor = float(vision_stale_factor)

    def run(self, blackboard: Blackboard) -> str:
        mission = bb_get(blackboard, "current_mission") or {}

        # RELEASE_ONLY test: skip the whole pick maneuver and go straight to the
        # truck-zone delivery (NAV_TO_TRUCK → RELEASE_LOAD) with the pretend QR.
        if mission.get("release_only"):
            logger.info("[PICK] release_only — skipping pick, going to NAV_TO_TRUCK.")
            return "picked"

        # --- 1) raise forks to the entry height BEFORE docking, so they are
        #         already staged when we centre on the pallet ---
        outcome = drive_lifter(
            self._debug, blackboard, self._publish_lifter,
            self._entry_level, self._lifter_timeout, tag="PICK entry",
        )
        if outcome == "stop":
            return "stop"
        if outcome == "timeout":
            return "failed"

        # --- 2) align onto the pallet (QR docking) ---
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

            # Alignment reached DONE → LATCH the QR decoded AT THE DOCK into
            # qr_value. At DONE the robot is close + centred on the pallet, so
            # this read is far more reliable than SEARCH's far-range/edge decode
            # (which can be garbled or empty). RELEASE_LOAD resolves the delivery
            # truck from qr_value, so a bad SEARCH read made it fail → IDLE. We
            # overwrite only with a non-empty dock read, and it then stays fixed
            # until the next PICK reaches DONE.
            qr_dock = bb_get(blackboard, "qr_detected")
            if qr_dock:
                prev = bb_get(blackboard, "qr_value")
                blackboard["qr_value"] = qr_dock
                if qr_dock != prev:
                    logger.info("[PICK] DONE — latched dock QR %r (SEARCH had %r).",
                                qr_dock, prev)
                else:
                    logger.info("[PICK] DONE — dock QR confirmed %r.", qr_dock)
            else:
                logger.warning("[PICK] DONE but no QR decoded at the dock — "
                               "RELEASE_LOAD will use SEARCH's qr_value %r.",
                               bb_get(blackboard, "qr_value"))

        # --- 3) close the final gap to the load. Roller (this base class) creeps
        #         with the VISION (Electric-80 logo) stop + wheel-stall/time
        #         fallbacks. PICK_FROM_RACK overrides _approach_load with a fixed
        #         odometry advance (the rack QR/logo leave the frame up close). ---
        if self._approach_load(blackboard) == "stop":
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
        # Reverse uses its OWN speed (faster than the forward creep): now loaded
        # with the pallet, the creep speed sometimes sat under the motor deadband
        # and the robot didn't move. No slam risk here — it's backing AWAY.
        if drive_for_time(self._debug, blackboard, self._publish_cmd,
                          -self._reverse_speed, 0.0, self._reverse_time,
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

        # Record the carry height so NAV_TO_TRUCK can RE-ASSERT it if the lifter
        # gets perturbed mid-route (dashboard button, a lifting_node restart).
        # Works for both this state and PICK_FROM_RACK (which inherits run()).
        blackboard["carry_lifter_level"] = self._transport_level

        return "done" if mission.get("pick_only") else "picked"

    # ------------------------------------------------------------------
    def _approach_load(self, blackboard: Blackboard) -> str:
        """Step 3: close the final gap to the load. Returns 'stop' on abort,
        any other string on success.

        Creep forward stopping by VISION (the Electric-80 logo at target
        distance) BEFORE contact, with wheel-stall + a time limit as safety
        fallbacks (brownout-safe). When ``center_kp`` > 0 the creep also CENTERS
        on the logo (steer on /approach_stop/center_error) so the lifter enters
        the pallet straight. PICK_FROM_RACK extends this with a small fixed
        odometry advance after the stop.
        """
        return drive_until_approach_stop(
            self._debug, blackboard, self._publish_cmd,
            self._drive_speed, 0.0, self._forward_time,
            grace=self._stall_grace, stall_speed=self._stall_speed,
            stall_ticks=self._stall_ticks,
            vision_enabled=self._vision_stop,
            vision_fresh_s=self._vision_fresh,
            tag="PICK fwd",
            center_kp=self._center_kp, center_deadband_px=self._center_deadband_px,
            center_w_max=self._center_w_max, center_fresh_s=self._center_fresh_s,
            vision_stale_slow_s=self._vision_stale_slow_s,
            vision_stale_factor=self._vision_stale_factor)
