"""PICK_FROM_RACK — mission step 2 for RACK_TO_TRUCK (mission 2).

Same maneuver as the roller PICK, with two differences selected by the state
NAME (`PICK_FROM_RACK`) + the lifter levels:

  * lifter levels (entry/lift/transport) from the pick_rack_* params;
  * perception swaps to its RACK profiles while /robot_state == PICK_FROM_RACK:
      - qr_quad_alignment → the rack QR DOCK calibration,
      - logo_stop_debug   → the rack LOGO template (foreshortened top-view logo);
  * the approach (inherited from Pick) is a logo-CENTERING creep — both roller
    and rack centre on the logo so the lifter enters straight; PickFromRack just
    adds a small fixed odometry advance after the logo-stop (the rack pallet
    sits a touch lower/closer than the logo-stop pose).
"""

from __future__ import annotations

from yasmin import Blackboard

from mission_control.states.pick import Pick
from mission_control.states._actions import drive_distance


class PickFromRack(Pick):
    """PICK for rack pallets: distinct state name (selects the rack QR + logo
    profiles in perception), rack lifter levels, the inherited logo-centering
    approach, plus a small fixed advance after the logo-stop.

    Extra kwargs over :class:`Pick`:
        advance_dist / advance_speed — the fixed odometry nudge after the logo-stop.
    (Centering gains center_kp/center_w_max flow through to :class:`Pick`.)
    """

    def __init__(self, *args, advance_dist: float = 0.01,
                 advance_speed: float = 0.04, **kwargs) -> None:
        kwargs.setdefault("state_name", "PICK_FROM_RACK")
        self._advance_dist = float(advance_dist)
        self._advance_speed = float(advance_speed)
        super().__init__(*args, **kwargs)

    def _approach_load(self, blackboard: Blackboard) -> str:
        """Pick's logo-centering creep, then a fixed odometry advance to the
        (slightly lower) rack pick pose. Returns 'stop' on abort, else 'ok'."""
        outcome = super()._approach_load(blackboard)
        if outcome == "stop":
            return "stop"
        return drive_distance(
            self._debug, blackboard, self._publish_cmd,
            self._advance_dist, self._advance_speed, tag="PICK_FROM_RACK adv")
