"""PICK_FROM_RACK — mission step 2 for RACK_TO_TRUCK (mission 2).

Same maneuver as the roller PICK (stage forks → QR-dock → approach → lift →
reverse → carry); what differs for a rack pallet is selected by the state NAME
(`PICK_FROM_RACK`) + the lifter levels, and the final approach:

  * lifter levels (entry/lift/transport) from the pick_rack_* params;
  * perception swaps to its RACK profiles while /robot_state == PICK_FROM_RACK:
      - qr_quad_alignment → the rack QR DOCK calibration (lower + farther QR),
      - logo_stop_debug   → the rack LOGO template (foreshortened top-view logo),
        publishing /approach_stop/should_stop + /approach_stop/center_error;
  * the approach (``_approach_load``) is a logo CENTERING creep: it steers on the
    logo's lateral error (like the dock centres on the QR) so the lifter enters
    the pallet straight, stops on the logo-stop, then a small fixed odometry
    advance (the rack pallet sits a touch lower/closer than the logo-stop pose).
"""

from __future__ import annotations

from yasmin import Blackboard

from mission_control.states.pick import Pick
from mission_control.states._actions import drive_until_approach_stop, drive_distance


class PickFromRack(Pick):
    """PICK for rack pallets: distinct state name (selects the rack QR + logo
    profiles in perception), rack lifter levels, and a logo-centering approach
    + a small fixed advance.

    Extra kwargs over :class:`Pick`:
        advance_dist / advance_speed — the fixed odometry nudge after the logo-stop.
        center_kp / center_w_max     — logo centering gain + turn cap for the creep.
    """

    def __init__(self, *args, advance_dist: float = 0.02, advance_speed: float = 0.04,
                 center_kp: float = 0.0025, center_w_max: float = 0.10,
                 center_deadband_px: float = 12.0, center_fresh_s: float = 1.0,
                 **kwargs) -> None:
        kwargs.setdefault("state_name", "PICK_FROM_RACK")
        self._advance_dist = float(advance_dist)
        self._advance_speed = float(advance_speed)
        self._center_kp = float(center_kp)
        self._center_w_max = float(center_w_max)
        self._center_deadband_px = float(center_deadband_px)
        self._center_fresh_s = float(center_fresh_s)
        super().__init__(*args, **kwargs)

    def _approach_load(self, blackboard: Blackboard) -> str:
        """Logo-CENTERING creep until the logo-stop fires, then a fixed odometry
        advance. Steering on /approach_stop/center_error keeps the pallet centred
        so the lifter enters straight. Returns 'stop' on abort, else 'ok'."""
        outcome = drive_until_approach_stop(
            self._debug, blackboard, self._publish_cmd,
            self._drive_speed, 0.0, self._forward_time,
            grace=self._stall_grace, stall_speed=self._stall_speed,
            stall_ticks=self._stall_ticks, vision_enabled=self._vision_stop,
            vision_fresh_s=self._vision_fresh, tag="PICK_FROM_RACK approach",
            center_kp=self._center_kp, center_deadband_px=self._center_deadband_px,
            center_w_max=self._center_w_max, center_fresh_s=self._center_fresh_s)
        if outcome == "stop":
            return "stop"
        return drive_distance(
            self._debug, blackboard, self._publish_cmd,
            self._advance_dist, self._advance_speed, tag="PICK_FROM_RACK adv")
