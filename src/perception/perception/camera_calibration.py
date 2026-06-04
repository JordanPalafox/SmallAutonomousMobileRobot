#!/usr/bin/env python3
"""Calibracion de camara con patron de tablero de ajedrez.

Suscribe a /cam_img, detecta esquinas internas del tablero y al acumular
suficientes capturas calcula la matriz intrinseca y los coeficientes de
distorsion. El resultado se guarda en el formato esperado por
`config/camera_params.yaml`.

Controles (sobre la ventana de OpenCV):
  SPACE  capturar el frame actual (solo si se detecta el tablero)
  c      calcular calibracion con las capturas acumuladas
  s      guardar YAML en la ruta de salida
  u      deshacer la ultima captura
  r      reiniciar capturas
  q/ESC  salir

Ejemplo:
  ros2 run perception camera_calibration \\
    --ros-args -p board_cols:=9 -p board_rows:=6 -p square_size:=0.025 \\
               -p output:=src/perception/config/camera_params.yaml
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image


class CameraCalibration(Node):
    def __init__(self) -> None:
        super().__init__('camera_calibration')

        # Parametros declarados
        self.declare_parameter('image_topic', '/cam_img')
        self.declare_parameter('qos', 'sensor_data')  # 'sensor_data' (best_effort) o 'reliable'
        self.declare_parameter('board_cols', 9)   # esquinas internas por fila
        self.declare_parameter('board_rows', 6)   # esquinas internas por columna
        self.declare_parameter('square_size', 0.025)  # metros
        self.declare_parameter('min_captures', 15)
        self.declare_parameter('output', 'src/perception/config/camera_params.yaml')
        self.declare_parameter('save_snapshots', True)
        self.declare_parameter('snapshot_dir', '/tmp/calib_snapshots')
        self.declare_parameter('marker_length', 0.05)  # se preserva en el YAML

        self.image_topic = self.get_parameter('image_topic').value
        self.board_size: Tuple[int, int] = (
            int(self.get_parameter('board_cols').value),
            int(self.get_parameter('board_rows').value),
        )
        self.square_size = float(self.get_parameter('square_size').value)
        self.min_captures = int(self.get_parameter('min_captures').value)
        self.output_path = str(self.get_parameter('output').value)
        self.save_snapshots = bool(self.get_parameter('save_snapshots').value)
        self.snapshot_dir = str(self.get_parameter('snapshot_dir').value)
        self.marker_length = float(self.get_parameter('marker_length').value)

        # Puntos 3D del tablero (z=0 en el plano del tablero)
        cols, rows = self.board_size
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.square_size
        self.objp = objp

        self.obj_points: List[np.ndarray] = []
        self.img_points: List[np.ndarray] = []
        self.image_size: Tuple[int, int] | None = None
        self.last_frame: np.ndarray | None = None
        self.last_corners: np.ndarray | None = None
        self.calibration_result: dict | None = None

        if self.save_snapshots:
            os.makedirs(self.snapshot_dir, exist_ok=True)

        self.bridge = CvBridge()

        qos_name = str(self.get_parameter('qos').value).lower()
        if qos_name in ('reliable', 'default'):
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                durability=DurabilityPolicy.VOLATILE,
            )
        else:  # sensor_data / best_effort
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                durability=DurabilityPolicy.VOLATILE,
            )

        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, qos)
        self.timer = self.create_timer(0.05, self.ui_loop)  # 20 Hz UI

        cv2.namedWindow('calibration', cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f'Esperando imagenes en {self.image_topic} | tablero={self.board_size} '
            f'cuadro={self.square_size*1000:.1f}mm | salida={self.output_path}'
        )

    # ------------------------------------------------------------------
    def image_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            try:
                raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                enc = (msg.encoding or '').lower()
                if enc in ('rgb8', 'rgba8'):
                    frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                elif enc in ('mono8', '8uc1'):
                    frame = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                elif enc.startswith('yuv'):
                    frame = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
                elif enc in ('bgra8',):
                    frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                else:
                    frame = raw if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"No pude convertir imagen (encoding={msg.encoding!r}): {exc}"
                )
                return
        self.last_frame = frame
        if self.image_size is None:
            self.image_size = (frame.shape[1], frame.shape[0])
            self.get_logger().info(
                f"Primer frame OK: {frame.shape[1]}x{frame.shape[0]} encoding={msg.encoding!r}"
            )

    # ------------------------------------------------------------------
    def detect_corners(self, frame: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        if not found:
            return None
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        return corners

    # ------------------------------------------------------------------
    def ui_loop(self) -> None:
        if self.last_frame is None:
            return
        display = self.last_frame.copy()
        self.last_corners = self.detect_corners(self.last_frame)
        if self.last_corners is not None:
            cv2.drawChessboardCorners(display, self.board_size, self.last_corners, True)
            status = 'OK - SPACE para capturar'
            color = (0, 255, 0)
        else:
            status = 'Tablero no detectado'
            color = (0, 0, 255)

        cv2.putText(display, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(
            display,
            f'capturas: {len(self.img_points)}/{self.min_captures}  '
            f"(c=calibrar  s=guardar  u=undo  r=reset  q=salir)",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
        cv2.imshow('calibration', display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            self.get_logger().info('Cerrando...')
            rclpy.shutdown()
        elif key == 32:  # SPACE
            self.capture()
        elif key == ord('c'):
            self.calibrate()
        elif key == ord('s'):
            self.save_yaml()
        elif key == ord('u'):
            self.undo()
        elif key == ord('r'):
            self.reset()

    # ------------------------------------------------------------------
    def capture(self) -> None:
        if self.last_corners is None or self.last_frame is None:
            self.get_logger().warn('No hay tablero detectado para capturar')
            return
        self.obj_points.append(self.objp.copy())
        self.img_points.append(self.last_corners.copy())
        idx = len(self.img_points)
        self.get_logger().info(f'Captura #{idx} agregada')
        if self.save_snapshots:
            path = os.path.join(self.snapshot_dir, f'capture_{idx:02d}.png')
            cv2.imwrite(path, self.last_frame)

    def undo(self) -> None:
        if not self.img_points:
            return
        self.obj_points.pop()
        self.img_points.pop()
        self.get_logger().info(f'Captura deshecha (quedan {len(self.img_points)})')

    def reset(self) -> None:
        self.obj_points.clear()
        self.img_points.clear()
        self.calibration_result = None
        self.get_logger().info('Capturas reiniciadas')

    # ------------------------------------------------------------------
    def calibrate(self) -> None:
        if self.image_size is None:
            self.get_logger().warn('Aun no recibo imagenes')
            return
        if len(self.img_points) < self.min_captures:
            self.get_logger().warn(
                f'Necesito al menos {self.min_captures} capturas '
                f'(tengo {len(self.img_points)})'
            )
            return
        self.get_logger().info('Calculando calibracion...')
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, self.image_size, None, None
        )

        # Error de reproyeccion promedio
        per_view_errors = []
        for objp, imgp, rvec, tvec in zip(self.obj_points, self.img_points, rvecs, tvecs):
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            err = cv2.norm(imgp, proj, cv2.NORM_L2) / len(proj)
            per_view_errors.append(err)
        mean_err = float(np.mean(per_view_errors))

        self.calibration_result = {
            'rms': float(rms),
            'mean_reprojection_error': mean_err,
            'camera_matrix': K,
            'distortion_coefficients': dist.flatten(),
            'image_size': self.image_size,
            'n_captures': len(self.img_points),
        }

        self.get_logger().info(
            f'OK | RMS={rms:.4f}px  reproj_mean={mean_err:.4f}px  '
            f'capturas={len(self.img_points)}'
        )
        self.get_logger().info(f'K=\n{K}')
        self.get_logger().info(f'dist={dist.flatten()}')
        self.get_logger().info("Presiona 's' para guardar el YAML")

    # ------------------------------------------------------------------
    def save_yaml(self) -> None:
        if self.calibration_result is None:
            self.get_logger().warn("Primero ejecuta calibrar (tecla 'c')")
            return

        K = self.calibration_result['camera_matrix']
        dist = self.calibration_result['distortion_coefficients']
        w, h = self.calibration_result['image_size']

        # Asegurar 5 coeficientes (k1, k2, p1, p2, k3) como en el YAML existente
        dist5 = list(dist[:5]) + [0.0] * max(0, 5 - len(dist))

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write('camera:\n')
            f.write(f'  image_width: {w}\n')
            f.write(f'  image_height: {h}\n')
            f.write('  # Row-major 3x3 intrinsic matrix [fx, 0, cx, 0, fy, cy, 0, 0, 1]\n')
            f.write(
                '  camera_matrix: ['
                + ', '.join(f'{v:.6f}' for v in K.flatten())
                + ']\n'
            )
            f.write('  # Distortion coefficients [k1, k2, p1, p2, k3]\n')
            f.write(
                '  distortion_coefficients: ['
                + ', '.join(f'{v:.6f}' for v in dist5)
                + ']\n'
            )
            f.write('  # Physical side length of ArUco markers in metres\n')
            f.write(f'  marker_length: {self.marker_length}\n')
            f.write(
                f'  # calibrated_at: {datetime.now().isoformat(timespec="seconds")} '
                f'rms={self.calibration_result["rms"]:.4f}px '
                f'captures={self.calibration_result["n_captures"]}\n'
            )

        self.get_logger().info(f'YAML guardado en {self.output_path}')


def main() -> None:
    rclpy.init()
    node = CameraCalibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
