#!/usr/bin/env python3
"""pose_jitter_probe — confirm (or rule out) LiDAR-driven heading jitter as the
cause of path-following weave, BEFORE changing any code.

Run it with the full stack up (Jetson SLAM + laptop odom/relay) while you drive
the robot MANUALLY with teleop.  It listens to the raw /slam_pose heading and
the smooth wheel-odom heading and reports, per 5 s window:

  * |dθ| per /slam_pose update  — the per-scan heading STEP of the raw AMCL
    estimate.  On a straight manual drive the odom heading is flat; if
    /slam_pose shows steps of a few degrees synced to scan arrivals, THAT is
    the jitter the controller chases.
  * projected ω twitch the steps would inject through the nav controller
    (P term kp·step, D term (step/dt)·kd after one filter tick) vs angular_max.
    If these approach angular_max=0.21 rad/s, the weave is pose-driven.
  * corr(|dθ_slam|, |ω|)  — the LATENCY signature.  If the steps grow with how
    fast you turn (corr ≳ 0.4), suspect scan_time_offset mis-tuning on the
    Jetson (project_lidar_latency) and fix THAT first — smoothing in nav would
    only paper over a latency error that also corrupts the map.
  * persistence  — fraction of steps that mean-revert within a few scans
    (jitter, safe to smooth) vs persist (a real localization correction the
    robot SHOULD honor; smoothing would lag it).

This script ONLY observes — it publishes nothing and touches no parameters.

Usage:
  source install/setup.bash
  python3 src/navigation/scripts/pose_jitter_probe.py
  # override topics/gains if needed:
  python3 src/navigation/scripts/pose_jitter_probe.py \
      --ros-args -p odom_topic:=/puzzlebot_controller/odom -p kd:=0.5
"""
from __future__ import annotations

import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def _yaw(z, w):
    return 2.0 * math.atan2(z, w)


def _wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan')
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-12 or syy <= 1e-12:
        return float('nan')
    return sxy / math.sqrt(sxx * syy)


def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[k]


class PoseJitterProbe(Node):
    def __init__(self):
        super().__init__('pose_jitter_probe')
        self._pose_topic = self.declare_parameter('pose_topic', '/slam_pose').value
        self._odom_topic = self.declare_parameter(
            'odom_topic', '/puzzlebot_controller/odom').value
        self._kp = float(self.declare_parameter('kp', 1.5).value)
        self._kd = float(self.declare_parameter('kd', 0.5).value)
        self._tau = float(self.declare_parameter('kd_tau', 0.20).value)
        self._report = float(self.declare_parameter('report_period', 5.0).value)

        self._prev_th = None
        self._prev_t = None
        self._last_omega = 0.0
        # rolling per-window samples
        self._steps = []          # |dθ| per update (rad)
        self._omegas = []         # |ω| at each update (rad/s)
        self._signed = deque(maxlen=8)  # recent signed steps, for persistence
        self._persist_hits = 0
        self._persist_total = 0
        self._win_start = None
        self._update_count = 0

        self.create_subscription(
            PoseStamped, self._pose_topic, self._on_pose, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, self._odom_topic, self._on_odom, qos_profile_sensor_data)
        self.create_timer(self._report, self._emit)

        self.get_logger().info(
            f'pose_jitter_probe: pose={self._pose_topic} odom={self._odom_topic} '
            f'| drive STRAIGHT with teleop; reports every {self._report:.0f}s.\n'
            f'  (also do a few gentle turns so the latency correlation is meaningful)')

    def _on_odom(self, msg: Odometry):
        self._last_omega = msg.twist.twist.angular.z

    def _on_pose(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        th = _yaw(msg.pose.orientation.z, msg.pose.orientation.w)
        if self._win_start is None:
            self._win_start = t
        if self._prev_th is not None:
            dth = _wrap(th - self._prev_th)
            dt = max(1e-3, t - self._prev_t)
            self._steps.append(abs(dth))
            self._omegas.append(abs(self._last_omega))
            self._update_count += 1
            # persistence: does this step reverse the previous one (jitter)
            # or continue it (real correction)?
            if self._signed:
                prev = self._signed[-1]
                if abs(dth) > 0.01:
                    self._persist_total += 1
                    if (dth > 0) == (prev > 0):
                        self._persist_hits += 1   # same direction → persists
            self._signed.append(dth)
            self._last_dt = dt
        self._prev_th = th
        self._prev_t = t

    def _emit(self):
        if not self._steps:
            self.get_logger().info('… no /slam_pose updates yet (is SLAM running?)')
            return
        steps_deg = sorted(math.degrees(s) for s in self._steps)
        mean_s = sum(steps_deg) / len(steps_deg)
        p95_s = _pct(steps_deg, 95)
        max_s = steps_deg[-1]
        rate = len(self._steps) / self._report
        corr = _pearson(self._steps, self._omegas)
        dt = getattr(self, '_last_dt', 0.1)

        # projected controller ω twitch from the p95 step (rad → rad/s)
        p95_rad = math.radians(p95_s)
        p_omega = self._kp * p95_rad
        raw_de = min(2.0, p95_rad / dt)               # nav clamps raw deriv to ±2
        alpha = dt / (self._tau + dt)
        d_omega = self._kd * (alpha * raw_de)
        omega_max = 0.21
        persist_frac = (self._persist_hits / self._persist_total
                        if self._persist_total else float('nan'))

        # quick verdicts
        jitter_drives = (p_omega + d_omega) >= 0.5 * omega_max
        latency_suspect = (not math.isnan(corr)) and corr >= 0.4
        mostly_jitter = (not math.isnan(persist_frac)) and persist_frac < 0.5

        self.get_logger().info(
            '\n──────── pose-jitter report ────────\n'
            f'  /slam_pose updates : {len(self._steps)}  (~{rate:.1f} Hz)\n'
            f'  per-update |dθ|    : mean {mean_s:.2f}°  p95 {p95_s:.2f}°  max {max_s:.2f}°\n'
            f'  proj. ω twitch     : P {p_omega:.3f} + D {d_omega:.3f} = '
            f'{p_omega + d_omega:.3f} rad/s   (angular_max={omega_max})\n'
            f'  corr(|dθ|,|ω|)     : {corr:+.2f}   '
            f'{"← grows with turn rate ⇒ LATENCY suspect" if latency_suspect else "(low ⇒ not latency-dominated)"}\n'
            f'  step persistence   : {persist_frac:.2f}  '
            f'{"(mostly mean-reverting ⇒ jitter, safe to smooth)" if mostly_jitter else "(persists ⇒ may be real corrections)"}\n'
            f'  VERDICT            : '
            f'{"POSE JITTER drives the weave" if jitter_drives else "pose jitter is small"}'
            f'{"  — but FIX SCAN LATENCY FIRST (scan_time_offset)" if latency_suspect else ""}\n'
            '  → If jitter-driven & low corr & mean-reverting: enable pose_source:=tf '
            '(+ relay tf_smoothing_alpha:=0.5 on laptop).\n'
            '────────────────────────────────────')
        # reset window
        self._steps.clear()
        self._omegas.clear()
        self._persist_hits = 0
        self._persist_total = 0


def main():
    rclpy.init()
    try:
        rclpy.spin(PoseJitterProbe())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
