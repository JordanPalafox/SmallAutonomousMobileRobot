"""Shared blocking action helpers used by mission states.

Each helper runs a 10 Hz poll loop that honours the DebugContext (abort + pause)
and returns a short outcome string, so state ``run()`` methods stay tiny and the
navigation / lifter / alignment logic lives in exactly one place (the old design
duplicated each of these across two near-identical states).
"""

from __future__ import annotations

import logging
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
    target = max(0, min(7, int(level)))
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
    deadline = time.monotonic() + timeout
    logger.info("[%s] alignment started (timeout=%.1fs)", tag, timeout)
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
            if time.monotonic() > deadline:
                blackboard["mission_error_reason"] = "alignment timeout"
                return "failed"

            time.sleep(POLL_INTERVAL)
    finally:
        publish_align(False)
