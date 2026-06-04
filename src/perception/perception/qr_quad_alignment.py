#!/usr/bin/env python3
"""Alineamiento al QR con maniobra de giro 90° + avance + DOCK visual.

Pipeline:
    IDLE    -> SEARCH (no det) | PLAN (det fresca)
    SEARCH  -> PLAN al detectar QR.
    PLAN
        Si |cx_px - target_cx_px| < center_threshold_px -> DOCK directo.
        Si no, decide lado por el signo del error pixel y entra a ROTATE.
    ROTATE
        Giro 90° closed-loop con odom hacia el lado del QR -> ADVANCE.
    ADVANCE
        Avance recto open-loop (con odom) advance_distance metros -> DOCK.
    DOCK
        Control pixel-space con v y w simultaneos hasta DONE.  Si pierde
        el QR, gira buscando hacia el lado opuesto a la maniobra previa.
    DONE / LOST

Activacion: SPACE en ventana o `/alignment_start true`.
"""
from __future__ import annotations

import math
import os
import time
from enum import Enum
from typing import List, Optional

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseArray, Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from .qr_pose_detector import QRPoseDetector
from .odometry_tracker import OdometryTracker


class State(str, Enum):
    IDLE = 'IDLE'
    SEARCH = 'SEARCH'
    PLAN = 'PLAN'
    ROTATE = 'ROTATE'
    ADVANCE = 'ADVANCE'
    DOCK = 'DOCK'
    DONE = 'DONE'
    LOST = 'LOST'


def _with_floor(cmd: float, floor: float, ceiling: float) -> float:
    """Si cmd!=0 pero |cmd|<floor, lo eleva al floor manteniendo signo.
    Luego clipea al rango [-ceiling, ceiling]."""
    if cmd == 0:
        return 0.0
    s = math.copysign(1.0, cmd)
    mag = max(floor, min(ceiling, abs(cmd)))
    return s * mag


def _load_camera_yaml(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    cam = data['camera']
    K = np.array(cam['camera_matrix'], dtype=np.float64).reshape(3, 3)
    dist = np.array(cam['distortion_coefficients'], dtype=np.float64).flatten()
    return K, dist


def compute_geometry(
    tvec: np.ndarray,
    corners: np.ndarray,
    cam_offset_x: float,
    cam_offset_y: float,
) -> dict:
    """Calcula la posicion del QR en frame robot + su centroide en pixeles.

    Devuelve un dict con: qr_x, qr_y, bearing_qr, dist_qr, cx_px, cy_px.
    """
    X_cam, _Y_cam, Z_cam = float(tvec[0]), float(tvec[1]), float(tvec[2])
    qr_x = cam_offset_x + Z_cam
    qr_y = cam_offset_y - X_cam
    bearing_qr = math.atan2(qr_y, qr_x)
    dist_qr = math.hypot(qr_x, qr_y)
    c = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    cx_px = float(c[:, 0].mean())
    cy_px = float(c[:, 1].mean())
    return {
        'qr_x': qr_x, 'qr_y': qr_y,
        'bearing_qr': bearing_qr,
        'dist_qr': dist_qr,
        'cx_px': cx_px, 'cy_px': cy_px,
    }


class QRQuadAlignmentNode(Node):
    def __init__(self) -> None:
        super().__init__('qr_quad_alignment')

        # ---- I/O ----
        self.declare_parameter('image_topic', '/video_source/raw')
        self.declare_parameter('qos', 'sensor_data')
        self.declare_parameter(
            'camera_params', 'src/perception/config/camera_params.yaml'
        )
        self.declare_parameter('marker_length', 0.030)
        self.declare_parameter('backend', 'auto')

        # ---- Geometria del setup ----
        self.declare_parameter('cam_offset_x', 0.100)
        self.declare_parameter('cam_offset_y', 0.0)
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_separation', 0.19)

        # ---- Threshold de la maniobra (pixel-space) ----
        # Si |cx_px - target_cx_px| < esto -> QR centrado, DOCK directo.
        # Si |cx_px - target_cx_px| >= esto -> maniobra 90° + DOCK.
        # Default 60 px en una imagen 320 ancho: el QR debe estar a mas
        # de 60 px del centro horizontal para que dispare la maniobra.
        self.declare_parameter('center_threshold_px', 60.0)
        # EMA sobre geometria para snapshot estable.
        self.declare_parameter('plan_ema_alpha', 0.30)

        # Velocidad angular del giro inicial de 90° hacia el lado del QR.
        self.declare_parameter('rotation_w', 0.30)
        # Despues del giro 90°, avanza esta distancia (metros) en linea
        # recta antes de pasar a DOCK. Sirve para acercarse a donde estaba
        # el QR originalmente.
        self.declare_parameter('advance_distance', 0.10)
        self.declare_parameter('advance_v', 0.07)

        # ---- DOCK (criterio pixel-space) ----
        # Posicion ideal del centro del QR en la imagen cuando terminar.
        # Defaults capturados con la camara actual (320x240): QR centrado
        # horizontalmente, cerca del borde inferior (justo antes de perderlo).
        self.declare_parameter('target_cx_px', 160.0)
        self.declare_parameter('target_cy_px', 190.0)
        self.declare_parameter('dock_tol_cx_px', 12.0)
        self.declare_parameter('dock_tol_cy_px', 15.0)
        # Ganancias DOCK
        self.declare_parameter('dock_max_linear', 0.035) # m/s — lento para no pasarse del target
        self.declare_parameter('kp_v_dock_px', 0.0006)   # m/s por pixel de err_cy
        self.declare_parameter('kp_w_dock_px', 0.0025)   # rad/s por pixel de err_cx
        self.declare_parameter('w_max', 0.08)            # cap suave
        # Ticks consecutivos centrados para confirmar DONE (a 10 Hz: 3 ~ 0.3s).
        self.declare_parameter('dock_done_stable_ticks', 3)

        # ---- DOCK forward criterion: DISTANCE mode (height-agnostic) ----
        # The pixel-cy criterion above only converges when the QR is BELOW the
        # camera axis (e.g. rack, low). For a QR ABOVE the axis (roller, high)
        # the QR rises in the image as we approach, so cy never reaches the
        # bottom target. dock_target_dist > 0 switches the forward axis to use
        # the measured QR distance (dist_qr) instead: advance until dist_qr <=
        # dock_target_dist. Works for any QR height. 0.0 = keep pixel-cy mode.
        self.declare_parameter('dock_target_dist', 0.0)   # m (0 = pixel-cy mode)
        self.declare_parameter('dock_dist_tol', 0.03)     # m tolerance for DONE
        self.declare_parameter('kp_v_dock_dist', 0.30)    # m/s per m of distance error
        # Losing the QR while already centred + near the target means we closed
        # in enough that the marker left the frame — that's a successful dock,
        # not a "search". After this many consecutive centred ticks, a QR loss
        # is treated as DONE (if also within dock_lost_margin of the target).
        self.declare_parameter('dock_lost_ticks', 2)
        self.declare_parameter('dock_lost_margin', 0.10)  # m extra distance allowed

        # ---- Vision / safety ----
        self.declare_parameter('detection_max_age_s', 3.0)
        self.declare_parameter('control_freq', 10.0)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('auto_start', False)
        self.declare_parameter('show_window', True)
        # Publish the annotated frame (QR detection + pose + state) so the web
        # dashboard can show it. Headless-friendly (no cv2 window required).
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('frame_id', 'camera_link')

        # ---- Leer ----
        self._image_topic   = str(self.get_parameter('image_topic').value)
        qos_name            = str(self.get_parameter('qos').value).lower()
        cam_path            = str(self.get_parameter('camera_params').value)
        self._marker_length = float(self.get_parameter('marker_length').value)
        backend             = str(self.get_parameter('backend').value)

        self._cam_off_x     = float(self.get_parameter('cam_offset_x').value)
        self._cam_off_y     = float(self.get_parameter('cam_offset_y').value)
        self._center_thr_px = float(self.get_parameter('center_threshold_px').value)
        self._plan_ema_a    = float(self.get_parameter('plan_ema_alpha').value)

        self._wheel_radius     = float(self.get_parameter('wheel_radius').value)
        self._wheel_separation = float(self.get_parameter('wheel_separation').value)

        self._rotation_w    = float(self.get_parameter('rotation_w').value)
        self._advance_dist  = float(self.get_parameter('advance_distance').value)
        self._advance_v     = float(self.get_parameter('advance_v').value)

        self._target_cx_px  = float(self.get_parameter('target_cx_px').value)
        self._target_cy_px  = float(self.get_parameter('target_cy_px').value)
        self._tol_cx_px     = float(self.get_parameter('dock_tol_cx_px').value)
        self._tol_cy_px     = float(self.get_parameter('dock_tol_cy_px').value)
        self._v_dock_max    = float(self.get_parameter('dock_max_linear').value)
        self._kp_v_dock_px  = float(self.get_parameter('kp_v_dock_px').value)
        self._kp_w_dock_px  = float(self.get_parameter('kp_w_dock_px').value)
        self._w_max         = float(self.get_parameter('w_max').value)
        self._dock_stable_n = int(self.get_parameter('dock_done_stable_ticks').value)
        self._dock_target_dist = float(self.get_parameter('dock_target_dist').value)
        self._dock_dist_tol    = float(self.get_parameter('dock_dist_tol').value)
        self._kp_v_dock_dist   = float(self.get_parameter('kp_v_dock_dist').value)
        self._dock_lost_ticks  = int(self.get_parameter('dock_lost_ticks').value)
        self._dock_lost_margin = float(self.get_parameter('dock_lost_margin').value)
        self._centered_count   = 0
        self._dock_stable_count: int = 0

        self._max_det_age   = float(self.get_parameter('detection_max_age_s').value)
        self._control_dt    = 1.0 / max(1e-3, float(self.get_parameter('control_freq').value))
        self._dry_run       = bool(self.get_parameter('dry_run').value)
        self._auto_start    = bool(self.get_parameter('auto_start').value)
        self._show_window   = bool(self.get_parameter('show_window').value)
        self._publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self._frame_id      = str(self.get_parameter('frame_id').value)

        # ---- Calibracion ----
        if not os.path.isabs(cam_path):
            cam_path = os.path.abspath(cam_path)
        if not os.path.exists(cam_path):
            self.get_logger().error(f'No existe camera_params: {cam_path}')
            raise SystemExit(1)
        K, dist = _load_camera_yaml(cam_path)
        self._K, self._dist = K, dist

        self._detector = QRPoseDetector(
            camera_matrix=K, dist_coeffs=dist,
            marker_length=self._marker_length, refine=True, backend=backend,
        )
        self.get_logger().info(
            f'Detector backend={self._detector.backend!r} marker={self._marker_length*1000:.1f}mm'
        )

        # ---- QoS ----
        if qos_name in ('reliable', 'default'):
            img_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST, depth=10,
                durability=DurabilityPolicy.VOLATILE,
            )
        else:
            img_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, depth=5,
                durability=DurabilityPolicy.VOLATILE,
            )
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---- ROS I/O ----
        self._bridge = CvBridge()
        self.create_subscription(Image, self._image_topic, self._image_cb, img_qos)
        self.create_subscription(Bool, '/alignment_start', self._start_cb, 10)
        self._pub_cmd = self.create_publisher(Twist, '/cmd_vel_in', cmd_qos)
        self._pub_state = self.create_publisher(String, '/alignment_state', 10)
        self._pub_qr = self.create_publisher(PoseArray, '/qr_poses', 10)
        self._pub_goal = self.create_publisher(Point, '/path_debug/goal', 10)
        # Mission control's SCAN_QR state subscribes here to learn which
        # pallet the robot is in front of. Only payloads that decoded
        # successfully are published (empty strings are dropped).
        self._pub_qr_id = self.create_publisher(String, '/qr_detected', 10)
        # Annotated debug image for the web dashboard camera view.
        self._pub_debug_img = self.create_publisher(Image, '/qr_quad_alignment/debug_image', 10)

        # ---- Odometria ----
        self._odom = OdometryTracker(
            self,
            wheel_radius=self._wheel_radius,
            wheel_separation=self._wheel_separation,
        )

        # ---- Estado interno ----
        self._state: State = State.IDLE if not self._auto_start else State.SEARCH
        self._geom_latest: Optional[dict] = None
        self._last_det_time: Optional[float] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_dets: List[dict] = []
        # Rotacion 90° aplicada en ROTATE (rad, +pi/2 o -pi/2 si hubo
        # maniobra, 0 si fue directo a DOCK). DOCK la usa para elegir la
        # direccion de busqueda cuando pierde el QR.
        self._plan_alpha: float = 0.0
        # Pose (x, y) del robot al iniciar ADVANCE, en plan-frame.
        self._advance_start_xy: tuple[float, float] = (0.0, 0.0)

        if self._show_window:
            cv2.namedWindow('qr_quad_alignment', cv2.WINDOW_NORMAL)

        self.create_timer(self._control_dt, self._control_tick)

        self.get_logger().info(
            f'qr_quad_alignment listo  '
            f'center_thr={self._center_thr_px:.0f}px  '
            f'target_px=({self._target_cx_px:.0f},{self._target_cy_px:.0f})  '
            f'dry_run={self._dry_run}'
        )

    # ------------------------------------------------------------------
    def _image_cb(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            # Fallback: passthrough y convertir manualmente segun encoding
            try:
                raw = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
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
            except Exception as exc2:  # noqa: BLE001
                self.get_logger().warn(
                    f'cv_bridge: encoding={msg.encoding!r} {exc2}',
                    throttle_duration_sec=2.0,
                )
                return
        if not hasattr(self, '_first_frame_logged'):
            self.get_logger().info(
                f'Primer frame {frame.shape[1]}x{frame.shape[0]} encoding={msg.encoding!r}'
            )
            self._first_frame_logged = True
        try:
            dets = self._detector.detect(frame)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'detect: {exc}', throttle_duration_sec=2.0)
            dets = []

        self._latest_frame = frame
        self._latest_dets = dets

        pa = PoseArray()
        pa.header.stamp = msg.header.stamp
        pa.header.frame_id = self._frame_id
        pa.poses = [d['pose'] for d in dets]
        self._pub_qr.publish(pa)

        # Publish the decoded payload of the closest QR (smallest tvec.z)
        # so the SM can identify which pallet is in front.
        decoded = [d for d in dets if d.get('id')]
        if decoded:
            closest = min(decoded, key=lambda d: float(d['tvec'][2]))
            id_msg = String()
            id_msg.data = str(closest['id'])
            self._pub_qr_id.publish(id_msg)

        if dets:
            d = dets[0]
            new_geom = compute_geometry(
                d['tvec'], d['corners'],
                self._cam_off_x, self._cam_off_y,
            )
            a = self._plan_ema_a
            if self._geom_latest is None:
                self._geom_latest = new_geom
            else:
                blended: dict = {}
                for k in ('qr_x', 'qr_y', 'dist_qr', 'cx_px', 'cy_px'):
                    blended[k] = (1 - a) * self._geom_latest[k] + a * new_geom[k]
                # bearing es angulo -> mezclar en seno/coseno
                s = (1 - a) * math.sin(self._geom_latest['bearing_qr']) + a * math.sin(new_geom['bearing_qr'])
                c = (1 - a) * math.cos(self._geom_latest['bearing_qr']) + a * math.cos(new_geom['bearing_qr'])
                blended['bearing_qr'] = math.atan2(s, c)
                self._geom_latest = blended
            self._last_det_time = time.monotonic()
            g = self._geom_latest
            p = Point(); p.x = g['qr_x']; p.y = g['qr_y']; p.z = 0.0
            self._pub_goal.publish(p)

    def _start_cb(self, msg: Bool) -> None:
        if msg.data:
            if self._is_detection_fresh() and self._geom_latest is not None:
                self._set_state(State.PLAN, reason='start request')
            else:
                self._set_state(State.SEARCH, reason='start, sin det')
        else:
            self._set_state(State.IDLE, reason='stop request')

    def _is_detection_fresh(self) -> bool:
        if self._last_det_time is None:
            return False
        return (time.monotonic() - self._last_det_time) <= self._max_det_age

    # ------------------------------------------------------------------
    def _control_tick(self) -> None:
        s = String(); s.data = self._state.value
        self._pub_state.publish(s)

        if self._state in (State.IDLE, State.DONE, State.LOST):
            # Idle/finished: stay OFF the /cmd_vel_in bus so navigation owns it.
            # We do NOT publish zero here every tick — that would fight nav_node
            # while the robot is driving to a candidate/truck. The one-shot
            # _publish_zero() in _set_state() already halts our own docking
            # motion when we leave an active state.
            self._render_ui()
            return
        if self._state == State.SEARCH:
            self._run_search()
            self._render_ui()
            return
        if self._state == State.PLAN:
            self._run_plan()
            self._render_ui()
            return
        if self._state == State.ROTATE:
            self._run_rotate()
            self._render_ui()
            return
        if self._state == State.ADVANCE:
            self._run_advance()
            self._render_ui()
            return
        if self._state == State.DOCK:
            self._run_dock()
            self._render_ui()
            return

    def _run_search(self) -> None:
        if self._is_detection_fresh() and self._geom_latest is not None:
            self._set_state(State.PLAN, reason='QR encontrado')
            return
        self._publish_cmd(0.0, 0.15)

    def _run_rotate(self) -> None:
        """Rotacion 90° closed-loop con odometria. Al terminar -> ADVANCE."""
        _, _, th_r = self._odom.pose()
        err = self._plan_alpha - th_r
        err = math.atan2(math.sin(err), math.cos(err))
        if abs(err) < 0.05:                              # ~3°
            self.get_logger().info(
                f'ROTATE completado: theta={math.degrees(th_r):+.1f}° '
                f'(target={math.degrees(self._plan_alpha):+.1f}°)'
            )
            # Snapshot la pose actual como origen para medir el avance.
            x_r, y_r, _ = self._odom.pose()
            self._advance_start_xy = (x_r, y_r)
            self._set_state(State.ADVANCE, reason='rotacion 90° completa')
            return
        w_raw = 2.0 * err
        w = _with_floor(w_raw, self._W_DEADBAND + 0.01, self._rotation_w)
        self._publish_cmd(0.0, w)

    def _run_advance(self) -> None:
        """Avance recto post-rotacion para acercar el robot al QR antes
        de DOCK. Usa odometria para medir la distancia recorrida.
        """
        x_r, y_r, _ = self._odom.pose()
        x0, y0 = self._advance_start_xy
        traveled = math.hypot(x_r - x0, y_r - y0)
        if traveled >= self._advance_dist:
            self.get_logger().info(
                f'ADVANCE completado: {traveled*1000:.0f}mm '
                f'(target={self._advance_dist*1000:.0f}mm)'
            )
            # Limpiar geom_latest: el QR cambio de lugar en la imagen
            # tras la rotacion + avance; forzar deteccion fresca en DOCK.
            self._geom_latest = None
            self._last_det_time = None
            self._set_state(State.DOCK, reason='avance completo')
            return
        self._publish_cmd(self._advance_v, 0.0)

    def _run_plan(self) -> None:
        """Decide la estrategia segun la posicion pixel del QR:
          * |cx_px - target_cx_px| < center_thr_px -> DOCK directo.
          * |cx_px - target_cx_px| >= center_thr_px -> ROTATE 90° hacia
            el lado del QR, luego DOCK con feedback visual.
        """
        if self._geom_latest is None:
            self.get_logger().warn('PLAN sin geom')
            self._set_state(State.LOST, reason='no geom en PLAN')
            return
        g = self._geom_latest
        err_cx = g['cx_px'] - self._target_cx_px

        if abs(err_cx) < self._center_thr_px:
            self.get_logger().info(
                f'QR centrado (|err_cx|={abs(err_cx):.0f}px < '
                f'{self._center_thr_px:.0f}px) -> DOCK directo'
            )
            self._odom.reset()
            self._plan_alpha = 0.0
            self._set_state(State.DOCK, reason='QR centrado, skip rotacion')
            return

        # Lado por el signo del error pixel: err_cx > 0 -> QR a la
        # DERECHA del target -> girar derecha. err_cx < 0 -> izquierda.
        side_sign = -1 if err_cx > 0 else +1
        side_name = 'izquierda' if side_sign > 0 else 'derecha'
        self._plan_alpha = side_sign * (math.pi / 2.0)
        self._odom.reset()

        self.get_logger().info(
            f'PLAN: maniobra-90° giro={side_name}  '
            f'err_cx={err_cx:+.0f}px'
        )
        self._set_state(State.ROTATE, reason='inicia rotacion 90°')

    def _near_target(self, g: dict) -> bool:
        """Whether the (possibly stale) geometry is within ~the dock target —
        used to decide if a QR loss means 'docked' rather than 'lost'."""
        if self._dock_target_dist > 0.0:
            return g['dist_qr'] <= self._dock_target_dist + self._dock_lost_margin
        # pixel-cy mode: QR has descended close to / past the target row
        return (g['cy_px'] - self._target_cy_px) >= -(self._tol_cy_px * 3.0)

    def _run_dock(self) -> None:
        """DOCK pixel-space con v y w SIMULTANEOS.

        En cada tick se calcula:
          v = kp_v * |err_cy|   (avance proporcional al deficit vertical)
          w = -kp_w * err_cx    (correccion lateral proporcional)
        Cuando el eje correspondiente esta dentro de tolerancia, el comando
        de ese eje se anula. Asi el robot avanza y se mantiene centrado al
        mismo tiempo, sin oscilar entre fases separadas.

          (a) SIN DETECCION -> gira buscando hacia lado opuesto a maniobra.
          (b) DONE: ambos ejes en tolerancia por N ticks estables.
        """
        # (a) sin deteccion -> gira buscando hacia el lado OPUESTO al de
        # la maniobra. Tras rotar 90° izquierda (plan_alpha=+pi/2), el QR
        # queda al lado derecho del nuevo heading -> buscar girando a la
        # derecha (search_dir=-1).
        if not (self._is_detection_fresh() and self._geom_latest is not None):
            # Lost the QR. If we were already centred and near the target, we
            # lost it BECAUSE we closed in (marker left the frame) → docked, DONE.
            g_last = self._geom_latest
            if (g_last is not None and self._centered_count >= self._dock_lost_ticks
                    and self._near_target(g_last)):
                self._set_state(
                    State.DONE,
                    reason=f'QR lost while docked (centered x{self._centered_count}, '
                           f'd={g_last["dist_qr"]*1000:.0f}mm)')
                self._centered_count = 0
                return
            # Genuinely lost → search by turning toward the last known side.
            if self._plan_alpha != 0.0:
                search_dir = -math.copysign(1.0, self._plan_alpha)
            else:
                if not hasattr(self, '_last_search_dir'):
                    self._last_search_dir = +1.0
                search_dir = self._last_search_dir
            self._dock_stable_count = 0
            self._centered_count = 0
            self._publish_cmd(0.0, 0.12 * search_dir)
            return

        g = self._geom_latest
        # err_cx > 0 -> QR a la DERECHA del target -> w<0 (girar derecha)
        # err_cy < 0 -> QR ARRIBA del target -> debe avanzar mas
        err_cx = g['cx_px'] - self._target_cx_px
        err_cy = g['cy_px'] - self._target_cy_px
        self._last_search_dir = -1.0 if err_cx >= 0 else +1.0

        cx_in_tol = abs(err_cx) < self._tol_cx_px
        self._centered_count = self._centered_count + 1 if cx_in_tol else 0
        # Forward axis: DISTANCE mode (height-agnostic) vs pixel-cy mode.
        if self._dock_target_dist > 0.0:
            dist_err = g['dist_qr'] - self._dock_target_dist
            fwd_in_tol = dist_err <= self._dock_dist_tol      # close enough
        else:
            dist_err = 0.0
            fwd_in_tol = err_cy >= -self._tol_cy_px           # cy: llegamos o pasamos

        # DONE: ambos en tolerancia por N ticks estables
        if cx_in_tol and fwd_in_tol:
            self._dock_stable_count += 1
            if self._dock_stable_count >= self._dock_stable_n:
                self._set_state(
                    State.DONE,
                    reason=f'centrado x{self._dock_stable_count} ticks: '
                           f'cx={g["cx_px"]:.0f}px (tg={self._target_cx_px:.0f}) '
                           f'cy={g["cy_px"]:.0f}px (tg={self._target_cy_px:.0f})'
                )
                return
            self._publish_cmd(0.0, 0.0)
            return

        # ---- Comando combinado v + w ----
        self._dock_stable_count = 0

        # v: solo si el eje de avance esta fuera de tolerancia
        if not fwd_in_tol:
            if self._dock_target_dist > 0.0:
                v_raw = self._kp_v_dock_dist * dist_err      # proporcional a la distancia restante
            else:
                v_raw = self._kp_v_dock_px * abs(err_cy)
            v = _with_floor(v_raw, self._V_DEADBAND + 0.005, self._v_dock_max)
        else:
            v = 0.0

        # w: solo si cx fuera de tolerancia (todavia tiene que centrar)
        if not cx_in_tol:
            w_raw = -self._kp_w_dock_px * err_cx
            w = _with_floor(w_raw, self._W_DEADBAND + 0.005, self._w_max)
        else:
            w = 0.0

        self._publish_cmd(v, w)

    # ------------------------------------------------------------------
    # Deadbands del motor (el smoother se encarga de acel/decel y el cap
    # de velocidad). LPF solo en w para suavizar.
    _V_DEADBAND = 0.015
    _W_DEADBAND = 0.03
    _CMD_LPF_ALPHA = 0.40

    def _publish_cmd(self, v: float, w: float) -> None:
        if self._dry_run:
            return
        v_safe = float(v)
        w_safe = float(w)
        if not hasattr(self, '_cmd_w_prev'):
            self._cmd_w_prev = 0.0
        # LPF SOLO en w (suaviza oscilaciones), v pasa directo
        a = self._CMD_LPF_ALPHA
        w_smooth = a * w_safe + (1.0 - a) * self._cmd_w_prev
        self._cmd_w_prev = w_smooth
        # Deadbands
        v_out = 0.0 if abs(v_safe) < self._V_DEADBAND else v_safe
        w_out = 0.0 if abs(w_smooth) < self._W_DEADBAND else w_smooth
        t = Twist()
        t.linear.x = v_out
        t.angular.z = w_out
        self._pub_cmd.publish(t)
        self.get_logger().info(
            f'cmd_vel v={v_out:+.3f} w={w_out:+.3f}  [{self._state.value}]',
            throttle_duration_sec=0.5,
        )

    def _publish_zero(self) -> None:
        if self._dry_run:
            return
        self._cmd_w_prev = 0.0
        zero = Twist()
        for _ in range(3):
            self._pub_cmd.publish(zero)

    def _set_state(self, new: State, reason: str = '') -> None:
        if new == self._state:
            return
        if new == State.DOCK:
            self._centered_count = 0   # fresh dock — don't carry over centering
        x_r, y_r, th_r = self._odom.pose()
        gs = (f' [odom=({x_r*1000:.0f},{y_r*1000:.0f},{math.degrees(th_r):+.0f})')
        if self._geom_latest is not None:
            g = self._geom_latest
            gs += (f' qr=({g["qr_x"]*1000:.0f},{g["qr_y"]*1000:.0f})'
                   f' bearing={math.degrees(g["bearing_qr"]):+.1f}°]')
        else:
            gs += ' no-qr]'
        self.get_logger().info(f'{self._state.value} -> {new.value} ({reason}){gs}')
        self._state = new
        self._publish_zero()

    # ------------------------------------------------------------------
    def _render_ui(self) -> None:
        if self._latest_frame is None:
            return
        # Build the annotated frame when EITHER a local window or the dashboard
        # debug-image stream is wanted (headless robot → publish only).
        if not (self._show_window or self._publish_debug_image):
            return
        out = self._detector.draw_detections(self._latest_frame, self._latest_dets)

        h, w = out.shape[:2]
        cv2.line(out, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
        cv2.line(out, (0, h // 2), (w, h // 2), (80, 80, 80), 1)

        # Target pixel del DOCK (cruz verde)
        cv2.drawMarker(out, (int(self._target_cx_px), int(self._target_cy_px)),
                       (0, 255, 0), cv2.MARKER_CROSS, 14, 2)

        state_color = {
            State.IDLE:    (200, 200, 200),
            State.SEARCH:  (255, 200,   0),
            State.PLAN:    (255, 128,   0),
            State.ROTATE:  (255,  80,   0),
            State.ADVANCE: (180, 100, 255),
            State.DOCK:    (  0, 200,   0),
            State.DONE:    (  0, 255,   0),
            State.LOST:    (  0,   0, 255),
        }[self._state]
        cv2.rectangle(out, (0, 0), (w, 22), state_color, -1)
        label = self._state.value + ('  [DRY]' if self._dry_run else '')
        cv2.putText(out, label, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        x_r, y_r, th_r = self._odom.pose()
        txts = [
            f"odom x={x_r*1000:+.0f} y={y_r*1000:+.0f} th={math.degrees(th_r):+.1f}",
        ]
        if self._geom_latest is not None:
            g = self._geom_latest
            fresh = self._is_detection_fresh()
            txts.append(f"QR x={g['qr_x']*1000:+.0f} y={g['qr_y']*1000:+.0f} d={g['dist_qr']*1000:.0f}mm fresh={fresh}")
        for i, t in enumerate(txts):
            cv2.putText(out, t, (6, 40 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

        # Publish the annotated frame so the dashboard camera shows the live QR
        # detection + pose (works headless, no cv2 window needed).
        if self._publish_debug_image:
            try:
                self._pub_debug_img.publish(self._bridge.cv2_to_imgmsg(out, encoding='bgr8'))
            except Exception:  # noqa: BLE001
                pass

        if not self._show_window:
            return
        cv2.imshow('qr_quad_alignment', out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            self._publish_zero()
            rclpy.shutdown()
        elif key == ord(' '):
            if self._state == State.IDLE:
                if self._is_detection_fresh() and self._geom_latest is not None:
                    self._set_state(State.PLAN, reason='SPACE')
                else:
                    self._set_state(State.SEARCH, reason='SPACE')
            else:
                self._set_state(State.IDLE, reason='SPACE')


def main() -> None:
    rclpy.init()
    node = QRQuadAlignmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish_zero()  # noqa: SLF001
        except Exception:
            pass
        cv2.destroyAllWindows()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
