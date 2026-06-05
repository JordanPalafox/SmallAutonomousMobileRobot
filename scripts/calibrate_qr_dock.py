#!/usr/bin/env python3
"""Capture the live QR-docking 'DONE' calibration at the current robot pose.

Hold the robot at the IDEAL dock pose (where alignment should be considered
DONE), then run this. It subscribes to the camera, runs the SAME
``QRPoseDetector`` + geometry that ``qr_quad_alignment`` uses, and reports the
MEDIAN ``target_cx_px`` / ``target_cy_px`` / ``dock_target_dist`` over N
detections — the three numbers that define the DONE criterion — ready to paste
into ``robot.launch.py``.

Read-only: it never publishes ``/cmd_vel`` so the robot does not move.

    python3 calibrate_qr_dock.py \
        --camera-params ~/puzzlebot_challenge_ws/src/perception/config/camera_params.yaml
"""
from __future__ import annotations

import argparse
import math
import time
from typing import List, Optional

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image

from perception.qr_pose_detector import QRPoseDetector


def _load_camera_yaml(path: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    cam = data['camera']
    K = np.array(cam['camera_matrix'], dtype=np.float64).reshape(3, 3)
    dist = np.array(cam['distortion_coefficients'], dtype=np.float64).flatten()
    w = int(cam.get('image_width', 0))
    h = int(cam.get('image_height', 0))
    return K, dist, w, h


class _Calib(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('qr_dock_calibrator')
        K, dist, cw, ch = _load_camera_yaml(args.camera_params)
        self._cw, self._ch = cw, ch
        self._args = args
        self._detector = QRPoseDetector(
            camera_matrix=K, dist_coeffs=dist,
            marker_length=args.marker_length, refine=True, backend='auto',
        )
        self._bridge = CvBridge()
        self._samples: list[dict] = []
        self._frames = 0
        self._res: Optional[tuple[int, int]] = None
        self._ids: set[str] = set()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, args.topic, self._cb, qos)
        self.get_logger().info(
            f'calibrator listo  topic={args.topic}  calib={cw}x{ch}  '
            f'marker={args.marker_length*1000:.0f}mm  backend={self._detector.backend!r}  '
            f'pyzbar={"ON" if self._detector.pyzbar_available else "OFF"}'
        )

    def _cb(self, msg: Image) -> None:
        self._frames += 1
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if self._res is None:
            self._res = (frame.shape[1], frame.shape[0])
            self.get_logger().info(
                f'LIVE frame {self._res[0]}x{self._res[1]} encoding={msg.encoding!r}'
                + ('' if self._res == (self._cw, self._ch)
                   else f'  *** != camera_params {self._cw}x{self._ch} ***')
            )
        try:
            dets = self._detector.detect(frame)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'detect: {exc}', throttle_duration_sec=2.0)
            return
        if not dets:
            return
        # Closest QR (smallest tvec.z) = the alignment target.
        d = min(dets, key=lambda x: float(x['tvec'][2]))
        tvec = d['tvec']
        corners = np.asarray(d['corners'], dtype=np.float64).reshape(-1, 2)
        X_cam, _Y, Z_cam = float(tvec[0]), float(tvec[1]), float(tvec[2])
        qr_x = self._args.cam_offset_x + Z_cam
        qr_y = self._args.cam_offset_y - X_cam
        self._samples.append({
            'cx_px': float(corners[:, 0].mean()),
            'cy_px': float(corners[:, 1].mean()),
            'dist_qr': math.hypot(qr_x, qr_y),
            'qr_x': qr_x, 'qr_y': qr_y,
            'bearing_deg': math.degrees(math.atan2(qr_y, qr_x)),
            'z_cam': Z_cam,
        })
        if d.get('id'):
            self._ids.add(str(d['id']))


def _stats(vals: list[float]) -> tuple[float, float, float]:
    a = np.asarray(vals, dtype=np.float64)
    return float(np.median(a)), float(np.percentile(a, 10)), float(np.percentile(a, 90))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--topic', default='/video_source/raw')
    ap.add_argument('--camera-params', required=True)
    ap.add_argument('--marker-length', type=float, default=0.05)
    ap.add_argument('--cam-offset-x', type=float, default=0.10)
    ap.add_argument('--cam-offset-y', type=float, default=0.0)
    ap.add_argument('--frames', type=int, default=50,
                    help='detections to collect before reporting')
    ap.add_argument('--timeout', type=float, default=30.0)
    args = ap.parse_args()

    rclpy.init()
    node = _Calib(args)
    t0 = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            if len(node._samples) >= args.frames:  # noqa: SLF001
                break
            if time.monotonic() - t0 > args.timeout:
                break
            if node._frames == 0 and (time.monotonic() - t0) > 3.0:  # noqa: SLF001
                print('   ...esperando frames (¿cámara publicando en este topic?)')
                t0 = time.monotonic() - 2.0  # re-arm the message ~1/s
    except KeyboardInterrupt:
        pass

    n = len(node._samples)  # noqa: SLF001
    print('=' * 64)
    print(f'frames vistos: {node._frames}   detecciones usadas: {n}')  # noqa: SLF001
    if node._res:  # noqa: SLF001
        print(f'resolucion viva: {node._res[0]}x{node._res[1]}'  # noqa: SLF001
              f'   (camera_params.yaml: {node._cw}x{node._ch})')  # noqa: SLF001
    if node._ids:  # noqa: SLF001
        print(f'QR id(s) leidos: {sorted(node._ids)}')  # noqa: SLF001
    if n < 5:
        print('\n[!] muy pocas detecciones. Revisa que el QR este en cuadro y a foco.')
        node.destroy_node(); rclpy.shutdown(); return

    s = node._samples  # noqa: SLF001
    cx_m, cx_lo, cx_hi = _stats([x['cx_px'] for x in s])
    cy_m, cy_lo, cy_hi = _stats([x['cy_px'] for x in s])
    d_m, d_lo, d_hi = _stats([x['dist_qr'] for x in s])
    qx_m, _, _ = _stats([x['qr_x'] for x in s])
    qy_m, _, _ = _stats([x['qr_y'] for x in s])
    br_m, _, _ = _stats([x['bearing_deg'] for x in s])
    z_m, _, _ = _stats([x['z_cam'] for x in s])

    print('\n--- MEDIANAS en la pose ideal (p10..p90) ---')
    print(f'  cx_px   = {cx_m:7.1f}   ({cx_lo:.1f}..{cx_hi:.1f})')
    print(f'  cy_px   = {cy_m:7.1f}   ({cy_lo:.1f}..{cy_hi:.1f})')
    print(f'  dist_qr = {d_m*1000:7.0f} mm ({d_lo*1000:.0f}..{d_hi*1000:.0f})  '
          f'[incluye cam_offset_x={args.cam_offset_x*1000:.0f}mm; z_cam={z_m*1000:.0f}mm]')
    print(f'  qr_x={qx_m*1000:.0f}mm  qr_y={qy_m*1000:.0f}mm  bearing={br_m:+.1f}deg')

    print('\n--- PEGAR en robot.launch.py (nodo qr_quad_alignment) ---')
    print(f"            'target_cx_px':        {cx_m:.1f},")
    print(f"            'target_cy_px':        {cy_m:.1f},")
    print('--- y el default del arg qr_dock_dist ---')
    print(f"        'qr_dock_dist', default_value='{d_m:.2f}',")
    print('=' * 64)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
