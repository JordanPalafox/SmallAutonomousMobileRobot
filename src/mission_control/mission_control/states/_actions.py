"""Shared blocking action helpers used by mission states.

Each helper runs a 10 Hz poll loop that honours the DebugContext (abort + pause)
and returns a short outcome string, so state ``run()`` methods stay tiny and the
navigation / lifter / alignment logic lives in exactly one place (the old design
duplicated each of these across two near-identical states).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable

from yasmin import Blackboard

from mission_control.debug_wrapper import DebugContext
from mission_control.bb_helpers import bb_get

logger = logging.getLogger(__name__)

POLL_INTERVAL = 0.1


def navigate(
    debug: DebugContext,
    blackboard: Blackboard,
    publish_goal: Callable[[str], None],
    target: str,
    *,
    tag: str,
) -> str:
    """Publish ``target`` on /goal_waypoint and poll /nav_status.

    Returns 'arrived' | 'error' | 'stop'. On abort the goal is cancelled.
    """
    blackboard["nav_status_prefix"] = "PLANNING"
    logger.info("[%s] → navigating to %s", tag, target)
    publish_goal(target)
    # Set once nav is actively working the goal. After that, a drop back to IDLE
    # means nav was cancelled out from under us (external stop, mode change, a
    # stale timer) WITHOUT arriving — we must bail, not poll forever, because the
    # velocity smoother holds nav's last command and would coast the robot on.
    seen_active = False
    active = ("FOLLOWING", "ALIGNING", "WALL_FOLLOWING", "RETURNING_LEAVE",
              "WAITING_FOR_CLEAR")
    while True:
        if debug.aborted:
            publish_goal("stop")
            return "stop"
        debug.wait_if_paused()

        prefix = bb_get(blackboard, "nav_status_prefix", "PLANNING")
        if prefix == "ARRIVED":
            logger.info("[%s] arrived at %s", tag, target)
            return "arrived"
        if prefix == "ERROR":
            blackboard["mission_error_reason"] = (
                f"navigation error at {target}: {bb_get(blackboard, 'nav_status_full')}"
            )
            return "error"
        if prefix in active:
            seen_active = True
        elif prefix == "IDLE" and seen_active:
            # Cancelled mid-route. Halt the robot (the smoother would otherwise
            # keep driving on nav's last command) and fail instead of hanging.
            publish_goal("stop")
            blackboard["mission_error_reason"] = (
                f"navigation to {target} was cancelled (nav returned to IDLE mid-route)"
            )
            logger.error("[%s] %s", tag, blackboard["mission_error_reason"])
            return "error"

        time.sleep(POLL_INTERVAL)


def drive_lifter(
    debug: DebugContext,
    blackboard: Blackboard,
    publish_lifter: Callable[[int], None],
    level: int,
    timeout: float,
    *,
    tag: str,
) -> str:
    """Command a lifter ``level`` and wait for /lifter_status to match.

    Returns 'done' | 'timeout' | 'stop'.
    """
    target = max(0, min(5, int(level)))
    logger.info("[%s] lifter → %d (timeout=%.1fs)", tag, target, timeout)
    publish_lifter(target)
    deadline = time.monotonic() + timeout
    while True:
        if debug.aborted:
            return "stop"
        debug.wait_if_paused()

        status = bb_get(blackboard, "lifter_status")
        if status is not None and int(status) == target:
            logger.info("[%s] lifter reached %d", tag, target)
            return "done"
        if time.monotonic() > deadline:
            blackboard["mission_error_reason"] = (
                f"lifter did not reach level {target} (last status={status})"
            )
            return "timeout"

        time.sleep(POLL_INTERVAL)


def drive_for_time(
    debug: DebugContext,
    blackboard: Blackboard,
    publish_cmd: Callable[[float, float], None],
    v: float,
    w: float,
    duration: float,
    *,
    tag: str,
) -> str:
    """Open-loop drive at (v, w) for ``duration`` seconds, then stop.

    Returns 'ok' | 'stop'. Honours abort/pause and always sends a zero command
    on exit. Used by PICK for the timed forward-into-pallet and back-out moves.
    """
    deadline = time.monotonic() + duration
    logger.info("[%s] drive v=%.3f w=%.3f for %.1fs", tag, v, w, duration)
    try:
        while time.monotonic() < deadline:
            if debug.aborted:
                return "stop"
            debug.wait_if_paused()
            publish_cmd(v, w)
            time.sleep(POLL_INTERVAL)
        return "ok"
    finally:
        publish_cmd(0.0, 0.0)


def drive_until_stall(
    debug: DebugContext,
    blackboard: Blackboard,
    publish_cmd: Callable[[float, float], None],
    v: float,
    w: float,
    max_duration: float,
    *,
    grace: float,
    stall_speed: float,
    stall_ticks: int,
    tag: str,
) -> str:
    """Drive at (v, w) until the wheels stall (robot blocked, e.g. pressed into
    the pallet) OR ``max_duration`` s elapses, then stop.

    Stall = wheel speed below ``stall_speed`` (rad/s, from blackboard['wheel_speed'])
    for ``stall_ticks`` consecutive ticks, only checked after a ``grace`` spin-up
    window so the initial ramp isn't mistaken for a stall. Returns 'stalled' |
    'timeout' | 'stop'. Both 'stalled' and 'timeout' mean the move finished
    normally — stalling against the pallet IS the success condition here, and
    stopping at once protects the motor driver from a long stall-current draw.
    """
    t0 = time.monotonic()
    deadline = t0 + max_duration
    grace_until = t0 + grace
    stalled = 0
    logger.info("[%s] drive v=%.3f for <=%.1fs (stop on wheel stall < %.2f rad/s)",
                tag, v, max_duration, stall_speed)
    try:
        while True:
            if debug.aborted:
                return "stop"
            debug.wait_if_paused()
            publish_cmd(v, w)

            now = time.monotonic()
            if now >= grace_until:
                ws = bb_get(blackboard, "wheel_speed")
                if ws is not None and abs(ws) < stall_speed:
                    stalled += 1
                    if stalled >= stall_ticks:
                        logger.info("[%s] wheels stalled (%.2f rad/s) — blocked, "
                                    "treating step as reached.", tag, ws)
                        return "stalled"
                else:
                    stalled = 0

            if now >= deadline:
                logger.info("[%s] reached time limit %.1fs.", tag, max_duration)
                return "timeout"
            time.sleep(POLL_INTERVAL)
    finally:
        publish_cmd(0.0, 0.0)


def drive_until_approach_stop(
    debug: DebugContext,
    blackboard: Blackboard,
    publish_cmd: Callable[[float, float], None],
    v: float,
    w: float,
    max_duration: float,
    *,
    grace: float,
    stall_speed: float,
    stall_ticks: int,
    vision_enabled: bool,
    vision_fresh_s: float,
    tag: str,
) -> str:
    """Creep into the load, stopping by VISION before contact, with the wheel
    stall and the time limit as safety fallbacks.

    This is the brownout fix for the PICK forward approach: the old path only
    stopped once the wheels stalled — i.e. once the robot had already crashed
    into the roller/pallet, and that stall-current spike is what browns out the
    Jetson's powerbank. Here the Electric-80 logo detector (perception/
    logo_stop_debug, on the Jetson) publishes /approach_stop/should_stop; the SM
    mirrors it into blackboard['approach_stop_signal'] as a (should_stop, stamp)
    tuple. We stop the instant that fires, BEFORE touching the load.

    Stop priority (all terminal outcomes mean the step succeeded):
      1. 'vision'  — approach_stop_signal True, fresher than ``vision_fresh_s``,
                     AND stamped after this creep began (only when
                     ``vision_enabled``). The brownout-safe path: stop before
                     contact.
      2. 'stalled' — wheels below ``stall_speed`` for ``stall_ticks`` ticks after
                     the ``grace`` spin-up. Fallback if the logo isn't seen
                     (detector off/occluded) — still ends the move on contact so
                     the driver isn't left stalling.
      3. 'timeout' — ``max_duration`` elapsed. Last-resort fallback.
      4. 'stop'    — abort raised.

    A stale True from a dead detector is ignored (freshness + stamp>=t0 gates) so
    we don't stop short on a frozen or pre-creep signal; with no fresh vision we
    degrade to the original stall/timeout behaviour. Always sends a zero on exit.
    """
    t0 = time.monotonic()
    deadline = t0 + max_duration
    grace_until = t0 + grace
    stalled = 0
    logger.info("[%s] creep v=%.3f for <=%.1fs (vision_stop=%s, fallback stall < "
                "%.2f rad/s)", tag, v, max_duration, vision_enabled, stall_speed)
    try:
        while True:
            if debug.aborted:
                return "stop"
            debug.wait_if_paused()
            publish_cmd(v, w)
            now = time.monotonic()

            # 1) Vision stop (primary) — stop BEFORE contact. Only honour a
            #    signal that ARRIVED AFTER this creep began (stamp >= t0): that
            #    ignores a stale True left over from the docking phase or a
            #    previous PICK (which would otherwise return on tick 1 without
            #    the robot ever moving), and requires a fresh confirmation that
            #    the logo is at target NOW. The Jetson twist_relay guard still
            #    cuts forward physically on any fresh True, so the few mm before
            #    that confirmation can't crash the robot.
            if vision_enabled:
                sig = bb_get(blackboard, "approach_stop_signal")
                if sig is not None:
                    should_stop, stamp = sig
                    if should_stop and stamp >= t0 and (now - stamp) <= vision_fresh_s:
                        logger.info("[%s] VISION stop — logo at target distance, "
                                    "halting before contact.", tag)
                        return "vision"

            # 2) Wheel stall (fallback) — only after the spin-up grace.
            if now >= grace_until:
                ws = bb_get(blackboard, "wheel_speed")
                if ws is not None and abs(ws) < stall_speed:
                    stalled += 1
                    if stalled >= stall_ticks:
                        logger.info("[%s] wheels stalled (%.2f rad/s) — blocked "
                                    "(vision fallback), step reached.", tag, ws)
                        return "stalled"
                else:
                    stalled = 0

            # 3) Time limit (fallback).
            if now >= deadline:
                logger.info("[%s] reached time limit %.1fs.", tag, max_duration)
                return "timeout"
            time.sleep(POLL_INTERVAL)
    finally:
        publish_cmd(0.0, 0.0)


def oscillate_until(
    debug: DebugContext,
    blackboard: Blackboard,
    publish_cmd: Callable[[float, float], None],
    predicate: Callable[[Blackboard], bool],
    *,
    angular_speed: float,
    sweep_start_deg: float,
    sweep_step_deg: float,
    sweep_max_deg: float,
    timeout: float,
    tag: str,
    progress_fn: Callable[[Blackboard], bool] | None = None,
) -> str:
    """Rotate IN PLACE (no translation), sweeping the heading back and forth
    with growing amplitude, until ``predicate(blackboard)`` is True.

    Used by RELEASE_LOAD to find a viewpoint where all three truck logos are in
    frame: from the arrival pose only two may be visible, so we turn a little
    right, a little left, widening each swing (sweep_start → sweep_max in
    sweep_step increments) until the logo detector reports all three.

    The heading is integrated open-loop from the commanded angular speed (no
    odometry feedback) — good enough to sweep the camera; the predicate is what
    actually stops the motion, so heading drift only affects the search pattern,
    not correctness. Linear velocity is always 0 (never advances/retreats).

    ``timeout`` is a NO-PROGRESS WATCHDOG when ``progress_fn`` is given: each tick
    that ``progress_fn(blackboard)`` is True (the search is still LEARNING — new
    logos/votes arriving) resets the deadline, so a slow-but-progressing scan
    isn't killed by a hard wall-clock cap. With no ``progress_fn`` it's a plain
    cap. Either way ``timeout`` <= 0 means "no deadline" (oscillate until found).

    Returns 'found' | 'timeout' | 'stop'. Always sends a zero command on exit.
    """
    t0 = time.monotonic()
    no_deadline = timeout <= 0.0          # <=0 ⇒ oscillate until found / abort
    deadline = None if no_deadline else t0 + timeout
    w_mag = abs(float(angular_speed))
    heading = 0.0                  # deg, open-loop estimate
    amp = float(sweep_start_deg)
    target = -amp                  # sweep to the right first
    going_negative = True
    last = t0
    logger.info("[%s] oscillating in place (sweep %.0f→%.0f° @ %.2f rad/s, timeout=%s) until target seen",
                tag, sweep_start_deg, sweep_max_deg, w_mag,
                "none" if no_deadline else f"{timeout:.1f}s")
    try:
        while True:
            if debug.aborted:
                return "stop"
            debug.wait_if_paused()

            if predicate(blackboard):
                logger.info("[%s] target acquired — stopping oscillation.", tag)
                return "found"

            now = time.monotonic()
            # Watchdog: while the search keeps making progress, push the deadline
            # out so a slow-but-working scan isn't cut off mid-resolve.
            if (progress_fn is not None and deadline is not None
                    and progress_fn(blackboard)):
                deadline = now + timeout
            if deadline is not None and now >= deadline:
                logger.warning("[%s] oscillation timed out (no progress for %.1fs).",
                               tag, timeout)
                return "timeout"

            dt = now - last
            last = now
            direction = 1.0 if (target - heading) > 0 else -1.0
            w = direction * w_mag
            publish_cmd(0.0, w)
            heading += math.degrees(w * dt)   # integrate the angle increment (deg)
            if abs(target - heading) <= 3.0:
                # Reached this extreme: flip side; grow amplitude after a full swing.
                if going_negative:
                    target = +amp
                    going_negative = False
                else:
                    amp = min(amp + sweep_step_deg, sweep_max_deg)
                    target = -amp
                    going_negative = True
            time.sleep(POLL_INTERVAL)
    finally:
        publish_cmd(0.0, 0.0)


def run_alignment(
    debug: DebugContext,
    blackboard: Blackboard,
    publish_align: Callable[[bool], None],
    timeout: float,
    *,
    tag: str,
) -> str:
    """Trigger qr_quad_alignment and wait for DONE / LOST / timeout.

    Always sends /alignment_start False on exit so the docking node returns to
    IDLE. Returns 'aligned' | 'failed' | 'stop'.
    """
    blackboard["alignment_state"] = "IDLE"
    publish_align(True)
    # `timeout` is a NO-PROGRESS WATCHDOG, not a hard wall-clock cap. The old
    # hard cap aborted a dock that was still converging — a SLOW-but-working
    # dock hit 30 s and got kicked to MISSION_FAILED → IDLE. Here the deadline
    # resets on any docking progress: a fresh QR decode (qr_detected_at advances
    # — published every frame the QR is in view, incl. during DOCK) or an
    # alignment-state change. So while the robot can see the QR it keeps docking
    # no matter how slow. A generous absolute ceiling still guarantees we give up
    # if it's genuinely stuck (QR lost / flickering forever without reaching DONE).
    start = time.monotonic()
    deadline = start + timeout
    hard_deadline = start + max(3.0 * timeout, timeout + 30.0)
    last_qr_at = bb_get(blackboard, "qr_detected_at", 0.0)
    last_state = None
    logger.info("[%s] alignment started (no-progress watchdog=%.1fs, ceiling=%.0fs)",
                tag, timeout, hard_deadline - start)
    try:
        while True:
            if debug.aborted:
                return "stop"
            debug.wait_if_paused()

            state = bb_get(blackboard, "alignment_state", "IDLE")
            if state == "DONE":
                logger.info("[%s] alignment DONE", tag)
                return "aligned"
            if state == "LOST":
                blackboard["mission_error_reason"] = "alignment LOST"
                return "failed"

            now = time.monotonic()
            qr_at = bb_get(blackboard, "qr_detected_at", 0.0)
            if qr_at != last_qr_at or state != last_state:
                # Progress: QR still seen, or the dock advanced a stage. Stay.
                last_qr_at = qr_at
                last_state = state
                deadline = now + timeout
            if now > deadline or now > hard_deadline:
                why = "no QR/progress" if now <= hard_deadline else "ceiling"
                blackboard["mission_error_reason"] = (
                    f"alignment stalled ({why}, state={state}, "
                    f"{now - start:.0f}s elapsed)"
                )
                logger.error("[%s] %s", tag, blackboard["mission_error_reason"])
                return "failed"

            time.sleep(POLL_INTERVAL)
    finally:
        publish_align(False)
