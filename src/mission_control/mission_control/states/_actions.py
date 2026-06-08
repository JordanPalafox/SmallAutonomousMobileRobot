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
              "WAITING_FOR_CLEAR", "WAITING_FOR_PATH")
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


def drive_distance(
    debug: DebugContext,
    blackboard: Blackboard,
    publish_cmd: Callable[[float, float], None],
    distance: float,
    speed: float,
    *,
    tag: str,
    max_time: float | None = None,
) -> str:
    """Drive straight a FIXED ``distance`` (m) at ``speed`` (m/s) using
    wheel-encoder odometry, then stop.

    Used by PICK_FROM_RACK for the final blind advance: at the rack pick pose the
    QR and logo are out of frame, so there is no vision feedback — but the
    rack↔pallet geometry is fixed, so a measured odometry advance is repeatable.

    Distance is integrated from ``blackboard['lin_vel']`` (signed linear speed
    derived from /wl,/wr by the SM node). If no encoder reading is available it
    falls back to integrating the COMMANDED speed (open-loop). A time cap
    (``max_time`` or ~2.5× the expected duration) guarantees termination even if
    the encoders go silent. ``distance`` < 0 drives backward. Returns 'ok' on
    completion (distance reached or cap hit) or 'stop' on abort. Always zeroes
    the command on exit.
    """
    target = abs(float(distance))
    spd = abs(float(speed)) * (1.0 if distance >= 0 else -1.0)
    cap = (float(max_time) if max_time is not None
           else target / max(1e-3, abs(float(speed))) * 2.5 + 2.0)
    traveled = 0.0
    last = time.monotonic()
    deadline = last + cap
    logger.info("[%s] odometry advance %.0f mm @ %.3f m/s (cap %.1fs)",
                tag, target * 1000.0, speed, cap)
    try:
        while traveled < target:
            if debug.aborted:
                return "stop"
            debug.wait_if_paused()
            now = time.monotonic()
            dt = now - last
            last = now
            v = bb_get(blackboard, "lin_vel")
            # Prefer measured encoder speed; fall back to the commanded speed if
            # no reading, so the advance still terminates by distance not only time.
            traveled += abs(v if v is not None else speed) * dt
            if now >= deadline:
                logger.warning("[%s] advance time cap (%.1fs) reached at %.0f mm.",
                               tag, cap, traveled * 1000.0)
                break
            publish_cmd(spd, 0.0)
            time.sleep(POLL_INTERVAL)
        logger.info("[%s] advance done: %.0f mm", tag, traveled * 1000.0)
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
    center_kp: float = 0.0,
    center_deadband_px: float = 10.0,
    center_w_max: float = 0.10,
    center_fresh_s: float = 1.0,
    vision_stale_slow_s: float = 0.0,
    vision_stale_factor: float = 1.0,
    vision_confirm: int = 1,
) -> str:
    """Creep into the load, stopping by VISION before contact, with the wheel
    stall and the time limit as safety fallbacks.

    Optional CENTERING (``center_kp`` > 0): instead of the constant ``w``,
    steer ``w = -center_kp * err_px`` to keep the pallet centred while creeping
    so the lifter enters straight. The error source is QR-PREFERRED: use
    blackboard['qr_center_error'] (px, from qr_quad_alignment) while the QR is
    still visible (fresher than ``center_fresh_s``), and fall back to
    blackboard['logo_center_error'] (px, from logo_stop_debug) once the QR has
    left the frame up close. Both carry the same sign convention, so the same
    gain applies; the command is capped at ``center_w_max`` and ignored within
    ``center_deadband_px``. ``center_kp`` = 0 keeps the constant ``w``.

    Camera-freeze safety (``vision_stale_factor`` < 1.0): if the vision
    heartbeat (the per-frame ``approach_stop`` signal) goes stale for longer
    than ``vision_stale_slow_s`` the feed has frozen and the robot is creeping
    blind, so the forward speed is scaled by ``vision_stale_factor`` (0 = hold)
    until the feed recovers — preventing an overshoot past the ideal stop while
    no fresh frame can fire the vision stop. Stall detection is suppressed while
    slowed so the reduced wheel speed isn't misread as contact.

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
    vision_hold = 0
    last_center_src = None
    stale_logged = False
    logger.info("[%s] creep v=%.3f for <=%.1fs (vision_stop=%s, fallback stall < "
                "%.2f rad/s)", tag, v, max_duration, vision_enabled, stall_speed)
    try:
        while True:
            if debug.aborted:
                return "stop"
            debug.wait_if_paused()
            now = time.monotonic()

            # Angular command: optional CENTERING (center_kp>0) overrides the
            # constant w — steer to bring the pallet to the frame centre so the
            # lifter enters straight. PREFER the QR lateral error while the QR is
            # still visible (fresh); fall back to the logo center error once the
            # QR has left the frame up close. Both errors share the same sign
            # convention (object_cx − target; +ve = right), so the same gain and
            # clamp apply to either source. Within the deadband or with no fresh
            # reading from either we drive straight.
            w_cmd = w
            if center_kp > 0.0:
                w_cmd = 0.0
                err_px = None
                src = None
                qe = bb_get(blackboard, "qr_center_error")
                if qe is not None:
                    q_err, q_stamp = qe
                    if (now - q_stamp) <= center_fresh_s:
                        err_px, src = q_err, "QR"
                if err_px is None:
                    ce = bb_get(blackboard, "logo_center_error")
                    if ce is not None:
                        l_err, l_stamp = ce
                        if (now - l_stamp) <= center_fresh_s:
                            err_px, src = l_err, "logo"
                if src != last_center_src:
                    if src is not None:
                        logger.info("[%s] centering on %s error.", tag, src)
                    last_center_src = src
                if err_px is not None and abs(err_px) > center_deadband_px:
                    w_cmd = max(-center_w_max, min(center_w_max, -center_kp * err_px))
            # Camera-freeze safety: /approach_stop/should_stop is republished on
            # EVERY processed camera frame, so its stamp is the vision heartbeat.
            # If it goes stale the feed has frozen / the detector stalled and the
            # robot is creeping BLIND toward the load — a frozen frame can sail it
            # past the ideal stop before the signal returns (the reported failure:
            # ~1 s freeze right at the ideal pose, no time to brake). So scale the
            # creep down while stale (vision_stale_factor: <1 slows, 0 holds) and
            # restore full speed when the feed recovers. Disabled at factor >= 1.0.
            sig = bb_get(blackboard, "approach_stop_signal") if vision_enabled else None
            v_eff = v
            slowed_for_stale = False
            if (vision_enabled and vision_stale_factor < 1.0 and sig is not None
                    and (now - sig[1]) > vision_stale_slow_s):
                v_eff = v * vision_stale_factor
                slowed_for_stale = True
                if not stale_logged:
                    logger.warning("[%s] vision feed STALE (>%.2f s, no camera "
                                   "update) — slowing creep to %.0f%% until it "
                                   "recovers (anti-overshoot).", tag,
                                   vision_stale_slow_s, vision_stale_factor * 100.0)
                    stale_logged = True
            elif stale_logged:
                logger.info("[%s] vision feed recovered — resuming full creep.", tag)
                stale_logged = False
            publish_cmd(v_eff, w_cmd)

            # 1) Vision stop (primary) — stop BEFORE contact. Only honour a
            #    signal that ARRIVED AFTER this creep began (stamp >= t0): that
            #    ignores a stale True left over from the docking phase or a
            #    previous PICK (which would otherwise return on tick 1 without
            #    the robot ever moving), and requires a fresh confirmation that
            #    the logo is at target NOW. The Jetson twist_relay guard still
            #    cuts forward physically on any fresh True, so the few mm before
            #    that confirmation can't crash the robot.
            if sig is not None:
                should_stop, stamp = sig
                if should_stop and stamp >= t0 and (now - stamp) <= vision_fresh_s:
                    vision_hold += 1
                    if vision_hold >= max(1, vision_confirm):
                        logger.info("[%s] VISION stop — logo at target (confirmed %d/%d).",
                                    tag, vision_hold, vision_confirm)
                        return "vision"
                else:
                    vision_hold = 0

            # 2) Wheel stall (fallback) — only after the spin-up grace, and NOT
            #    while we're deliberately slowing for a stale feed (the lower
            #    wheel speed would otherwise be misread as contact → false stop
            #    at the wrong pose).
            if now >= grace_until and not slowed_for_stale:
                ws = bb_get(blackboard, "wheel_speed")
                if ws is not None and abs(ws) < stall_speed:
                    stalled += 1
                    if stalled >= stall_ticks:
                        logger.info("[%s] wheels stalled (%.2f rad/s) — blocked "
                                    "(vision fallback), step reached.", tag, ws)
                        return "stalled"
                else:
                    stalled = 0
            elif slowed_for_stale:
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
