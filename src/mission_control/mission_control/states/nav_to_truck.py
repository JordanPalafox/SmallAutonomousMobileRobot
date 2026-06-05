"""NAV_TO_TRUCK — mission step 3: drive to the truck zone.

Picks the truck waypoint, then drives there. Destination selection, in order
of precedence:

    1. ``mission['destination']`` — an explicit waypoint (CUSTOM missions).
    2. The QR→truck alias (``qr_aliases`` in zones.yaml), but ONLY when
       ``resolve_from_qr`` is enabled — this is the future "deliver the pallet
       to the truck its QR encodes" behaviour.
    3. ``default_truck`` (``truck_default_waypoint`` param, default ``truck_1``).

For the roller/rack→truck flow the robot just goes to ``truck_1`` (the vantage
point): ``resolve_from_qr`` is off and no explicit destination is set, so it
falls back to the default truck. Flip ``truck_resolve_from_qr`` to true later to
route per-QR.

On arrival it always returns ``arrived`` → RELEASE_LOAD, which performs the
actual delivery (match the truck logo to the QR, drive there, drop the pallet).
A CUSTOM mission with an explicit destination is driven straight to it here, and
RELEASE_LOAD then just releases in place.
"""

from __future__ import annotations

import logging
from typing import Callable

from yasmin import Blackboard

from mission_control.debug_wrapper import DebuggableState, DebugContext
from mission_control.bb_helpers import bb_get
from mission_control.mission_parser import resolve_qr_to_waypoint
from mission_control.states._actions import navigate, drive_lifter

logger = logging.getLogger(__name__)


class NavToTruck(DebuggableState):
    """Step 3 — *navega a la zona de camiones*.

    Outcomes:
        arrived — reached the truck zone → RELEASE_LOAD (match logo & deliver).
        failed  — navigation error (no path / unknown waypoint).
        stop    — abort raised.
    """

    def __init__(
        self,
        debug_ctx: DebugContext,
        publish_goal_fn: Callable[[str], None],
        publish_lifter_fn: Callable[[int], None],
        zones_data: dict,
        lifter_timeout: float = 8.0,
        default_truck: str = "truck_1",
        resolve_from_qr: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            "NAV_TO_TRUCK", ["arrived", "failed", "stop"], debug_ctx,
            abort_outcome="stop", **kwargs,
        )
        self._publish_goal = publish_goal_fn
        self._publish_lifter = publish_lifter_fn
        self._lifter_timeout = float(lifter_timeout)
        self._zones = zones_data
        self._default_truck = str(default_truck)
        self._resolve_from_qr = bool(resolve_from_qr)

    def run(self, blackboard: Blackboard) -> str:
        mission = bb_get(blackboard, "current_mission") or {}

        # 0) RE-ASSERT the carry height before driving off. The lifter is NOT
        # commanded anywhere between PICK and RELEASE_LOAD, so if it got perturbed
        # (the dashboard lifter button, or a lifting_node restart — the Jetson GPIO
        # HAL boots all pins LOW = level 0) the pallet would coast at the wrong
        # height all the way to the truck. PICK records the height it left under
        # carry_lifter_level; we re-command it here and wait for confirmation so the
        # forks are back up BEFORE we move. A lifter timeout here is non-fatal (we
        # still navigate) — it only means the status echo was slow, not that the
        # mission failed. Skipped when no pick ran (RELEASE_ONLY: no pallet).
        carry = bb_get(blackboard, "carry_lifter_level")
        if carry is not None:
            outcome = drive_lifter(
                self._debug, blackboard, self._publish_lifter,
                int(carry), self._lifter_timeout, tag="NAV_TO_TRUCK carry",
            )
            if outcome == "stop":
                return "stop"
            if outcome == "timeout":
                logger.warning(
                    "[NAV_TO_TRUCK] lifter did not confirm carry level %s; "
                    "navigating anyway.", carry,
                )

        # 1) An explicit destination (CUSTOM) always wins.
        dest = mission.get("destination")

        # 2) Optionally route per-QR via the qr_aliases mapping.
        if not dest and self._resolve_from_qr:
            qr = bb_get(blackboard, "qr_value")
            resolved = resolve_qr_to_waypoint(self._zones, qr or "")
            if resolved:
                logger.info("[NAV_TO_TRUCK] QR %r → %s", qr, resolved)
                dest = resolved
            else:
                logger.warning(
                    "[NAV_TO_TRUCK] QR %r not in qr_aliases — using default truck %s.",
                    qr, self._default_truck,
                )

        # 3) Default truck (mission 1: always lands here → truck_1).
        if not dest:
            dest = self._default_truck
            logger.info("[NAV_TO_TRUCK] → default truck %s", dest)

        blackboard["resolved_dest"] = dest

        outcome = navigate(self._debug, blackboard, self._publish_goal, dest, tag="NAV_TO_TRUCK")
        if outcome == "arrived":
            return "arrived"   # → RELEASE_LOAD (match logo to QR, deliver, drop)
        if outcome == "stop":
            return "stop"
        return "failed"   # nav 'error'
