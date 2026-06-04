#!/usr/bin/env python3
"""Capa de localizacion global por marcadores ArUco en el piso.

Idea: los ~20 ArUco son puntos de referencia fijos con ID unico, pegados
PLANOS en el piso (mirando hacia arriba). Conociendo donde esta cada uno
(mapa `aruco_map.yaml`) y midiendo su pose con la camara, el robot deduce
su propia pose absoluta en la pista.

Pipeline por frame:
    1. Detecta markers en `/cam_img` -> pose marker en frame OPTICO de la camara.
    2. Para cada marker con ID conocido en el mapa, compone transformaciones:
           T_map_base = T_map_marker . inv(T_cam_marker) . inv(T_base_cam)
       y extrae (x, y, yaw) del robot en el mapa.
    3. Descarta detecciones lejanas/ruidosas, fusiona las restantes (media
       ponderada por 1/dist^2; yaw por media circular).
    4. Publica `/aruco_pose_estimate` (PoseWithCovarianceStamped, frame `map`)
       como MEDICION para el EKF del paquete `localization`.

Frames (REP-103): map y base_link con X adelante, Y izquierda, Z arriba.
Frame OPTICO de camara: Z adelante (sale del lente), X derecha, Y abajo.

EXTRINSECOS DE CAMARA (cam_xyz / cam_pitch_deg): son medidas FISICAS del
montaje y hay que ajustarlas al robot real. El URDF monta la camara en
xyz=(0.1, 0, 0.02) sobre base_link (~0.07 m del piso) mirando al frente;
para ver markers en el piso necesita inclinacion hacia abajo (cam_pitch_deg).
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

from .aruco_detector import ArucoDetector
from .image_decode import imgmsg_to_bgr


# ---------------------------------------------------------------------------
# Helpers de transformaciones homogeneas 4x4
# ---------------------------------------------------------------------------
def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _make_tf(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).flatten()
    return T


def _inv_tf(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _yaw_of(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def _load_camera_yaml(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    cam = data['camera']
    K = np.array(cam['camera_matrix'], dtype=np.float64).reshape(3, 3)
    dist = np.array(cam['distortion_coefficients'], dtype=np.float64).flatten()
    return K, dist


def _load_aruco_map(path: str) -> Dict[int, np.ndarray]:
    """Lee aruco_map.yaml -> {id: T_map_marker (4x4)}.

    Cada marker esta PLANO en el piso (z=0): su frame tiene Z = +Z del mapa
    (arriba) y theta = giro alrededor de Z del mapa. Acepta theta en grados
    (theta_deg) o radianes (theta).
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    markers = data.get('markers', data) if isinstance(data, dict) else data
    # Acepta lista [{id, x, y, theta_deg}, ...] o dict {id: {x, y, ...}, ...}.
    if isinstance(markers, dict):
        items = [dict(m, id=m.get('id', k)) for k, m in markers.items()]
    else:
        items = list(markers)
    out: Dict[int, np.ndarray] = {}
    for m in items:
        mid = int(m['id'])
        x = float(m['x'])
        y = float(m['y'])
        z = float(m.get('z', 0.0))
        if 'theta_deg' in m:
            theta = math.radians(float(m['theta_deg']))
        else:
            theta = float(m.get('theta', 0.0))
        out[mid] = _make_tf(_rot_z(theta), [x, y, z])
    return out


class ArucoLocalizationNode(Node):
    def __init__(self) -> None:
        super().__init__('aruco_localization')

        # ---- I/O ----
        self.declare_parameter('image_topic', '/cam_img')
        self.declare_parameter('qos', 'sensor_data')
        self.declare_parameter('camera_params', 'src/perception/config/camera_params.yaml')
        self.declare_parameter('aruco_map', 'src/perception/config/aruco_map.yaml')
        self.declare_parameter('marker_length', 0.09)
        self.declare_parameter('dictionary', 'original')
        self.declare_parameter('map_frame', 'map')

        # ---- Extrinsecos camara (frame OPTICO en base_link) ----
        # cam_xyz: origen del frame optico de la camara en base_link [m].
        # cam_pitch_deg: inclinacion hacia ABAJO de la camara [grados].
        #   AJUSTAR al montaje real; sin tilt la camara no ve el piso.
        self.declare_parameter('cam_xyz', [0.10, 0.0, 0.07])
        self.declare_parameter('cam_pitch_deg', 30.0)
        self.declare_parameter('cam_yaw_deg', 0.0)   # desalineacion lateral, normalmente 0

        # ---- Filtrado / fusion ----
        self.declare_parameter('max_range', 1.5)        # m; descarta markers mas lejanos
        self.declare_parameter('min_markers', 1)        # detecciones minimas para publicar
        self.declare_parameter('xy_std_base', 0.03)     # m, ruido base a 1 m
        self.declare_parameter('yaw_std_base_deg', 3.0)  # grados, ruido base a 1 m

        # ---- Tambien publicar /aruco_poses (para no correr 2 nodos) ----
        self.declare_parameter('publish_poses', True)
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('publish_debug_image', True)

        image_topic = str(self.get_parameter('image_topic').value)
        qos_name = str(self.get_parameter('qos').value).lower()
        cam_path = str(self.get_parameter('camera_params').value)
        map_path = str(self.get_parameter('aruco_map').value)
        marker_length = float(self.get_parameter('marker_length').value)
        dictionary = str(self.get_parameter('dictionary').value)
        self._map_frame = str(self.get_parameter('map_frame').value)

        cam_xyz = [float(v) for v in self.get_parameter('cam_xyz').value]
        cam_pitch = math.radians(float(self.get_parameter('cam_pitch_deg').value))
        cam_yaw = math.radians(float(self.get_parameter('cam_yaw_deg').value))

        self._max_range = float(self.get_parameter('max_range').value)
        self._min_markers = int(self.get_parameter('min_markers').value)
        self._xy_std_base = float(self.get_parameter('xy_std_base').value)
        self._yaw_std_base = math.radians(float(self.get_parameter('yaw_std_base_deg').value))

        self._publish_poses = bool(self.get_parameter('publish_poses').value)
        self._camera_frame = str(self.get_parameter('camera_frame').value)
        self._publish_debug = bool(self.get_parameter('publish_debug_image').value)

        # ---- Extrinseco base_link -> camara optica ----
        # Frame optico sin tilt respecto a base_link (Z=adelante, X=derecha,
        # Y=abajo): R = [[0,0,1],[-1,0,0],[0,-1,0]].
        # Se anteponen un pitch (abajo, sobre Y de base) y un yaw (sobre Z de base).
        R_body_opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
        R_base_cam = _rot_z(cam_yaw) @ _rot_y(cam_pitch) @ R_body_opt
        self._T_base_cam = _make_tf(R_base_cam, cam_xyz)

        # ---- Calibracion + mapa ----
        cam_path = cam_path if os.path.isabs(cam_path) else os.path.abspath(cam_path)
        map_path = map_path if os.path.isabs(map_path) else os.path.abspath(map_path)
        for p, label in ((cam_path, 'camera_params'), (map_path, 'aruco_map')):
            if not os.path.exists(p):
                self.get_logger().error(f'No existe {label}: {p}')
                raise SystemExit(1)
        K, dist = _load_camera_yaml(cam_path)
        self._map = _load_aruco_map(map_path)

        self._detector = ArucoDetector(
            camera_matrix=K, dist_coeffs=dist,
            marker_length=marker_length, dictionary=dictionary, refine=True,
        )
        self.get_logger().info(
            f'aruco_localization: {len(self._map)} markers en mapa, '
            f'dict={self._detector.dictionary}, marker={marker_length * 100:.1f}cm, '
            f'cam_pitch={math.degrees(cam_pitch):.0f}deg'
        )

        # ---- QoS / I/O ----
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

        self._bridge = CvBridge()
        self.create_subscription(Image, image_topic, self._image_cb, img_qos)
        self._pub_est = self.create_publisher(
            PoseWithCovarianceStamped, '/aruco_pose_estimate', 10)
        self._pub_poses = (
            self.create_publisher(PoseArray, '/aruco_poses', 10)
            if self._publish_poses else None)
        self._pub_ids = (
            self.create_publisher(Int32MultiArray, '/aruco_ids', 10)
            if self._publish_poses else None)
        self._pub_debug = (
            self.create_publisher(Image, '/aruco_localization/debug_image', 10)
            if self._publish_debug else None)

        self.get_logger().info(f'aruco_localization listo en {image_topic}')

    # ------------------------------------------------------------------
    def _robot_pose_from_marker(self, det: dict) -> Optional[tuple[float, float, float, float]]:
        """Devuelve (x, y, yaw, dist) del robot en el mapa, o None si el ID
        no esta en el mapa."""
        T_map_marker = self._map.get(int(det['id']))
        if T_map_marker is None:
            return None
        R_cm, _ = cv2.Rodrigues(np.asarray(det['rvec'], dtype=np.float64).reshape(3, 1))
        T_cam_marker = _make_tf(R_cm, det['tvec'])
        T_map_base = T_map_marker @ _inv_tf(T_cam_marker) @ _inv_tf(self._T_base_cam)
        x = float(T_map_base[0, 3])
        y = float(T_map_base[1, 3])
        yaw = _yaw_of(T_map_base[:3, :3])
        dist = float(np.linalg.norm(np.asarray(det['tvec']).flatten()))
        return x, y, yaw, dist

    def _image_cb(self, msg: Image) -> None:
        frame = imgmsg_to_bgr(self._bridge, msg)
        if frame is None:
            return
        dets = self._detector.detect(frame)

        if self._pub_poses is not None:
            self._publish_raw_poses(dets, msg)

        # ---- Estimaciones por marker conocido ----
        xs, ys, yaws, weights = [], [], [], []
        used = 0
        for det in dets:
            est = self._robot_pose_from_marker(det)
            if est is None:
                continue
            x, y, yaw, dist = est
            if dist > self._max_range or dist <= 1e-3:
                continue
            w = 1.0 / (dist * dist)
            xs.append(x); ys.append(y); yaws.append(yaw); weights.append(w)
            used += 1

        if self._pub_debug is not None:
            annotated = self._detector.draw_detections(frame, dets)
            out = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out.header.stamp = msg.header.stamp
            out.header.frame_id = self._camera_frame
            self._pub_debug.publish(out)

        if used < self._min_markers:
            return

        w = np.asarray(weights)
        wsum = float(w.sum())
        x_f = float(np.dot(w, xs) / wsum)
        y_f = float(np.dot(w, ys) / wsum)
        # Media circular para yaw (ponderada).
        s = float(np.dot(w, np.sin(yaws)))
        c = float(np.dot(w, np.cos(yaws)))
        yaw_f = math.atan2(s, c)

        # Incertidumbre: el ruido base es a 1 m; escala ~ dist_efectiva / sqrt(N).
        dist_eff = math.sqrt(1.0 / (wsum / used))  # ~ RMS de la distancia usada
        n = float(used)
        xy_var = (self._xy_std_base * dist_eff) ** 2 / n
        yaw_var = (self._yaw_std_base * dist_eff) ** 2 / n

        self._publish_estimate(x_f, y_f, yaw_f, xy_var, yaw_var, msg.header.stamp)
        self.get_logger().debug(
            f'pose=({x_f:.3f},{y_f:.3f},{math.degrees(yaw_f):.1f}deg) N={used}')

    # ------------------------------------------------------------------
    def _publish_raw_poses(self, dets, msg: Image) -> None:
        pa = PoseArray()
        pa.header.stamp = msg.header.stamp
        pa.header.frame_id = self._camera_frame
        ids = Int32MultiArray()
        for det in dets:
            pa.poses.append(det['pose'])
            ids.data.append(int(det['id']))
        self._pub_poses.publish(pa)
        self._pub_ids.publish(ids)

    def _publish_estimate(self, x, y, yaw, xy_var, yaw_var, stamp) -> None:
        m = PoseWithCovarianceStamped()
        m.header.stamp = stamp
        m.header.frame_id = self._map_frame
        m.pose.pose.position.x = x
        m.pose.pose.position.y = y
        m.pose.pose.position.z = 0.0
        m.pose.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.pose.orientation.w = math.cos(yaw / 2.0)
        # Covarianza 6x6 row-major (x, y, z, roll, pitch, yaw).
        cov = [0.0] * 36
        cov[0] = xy_var          # x
        cov[7] = xy_var          # y
        cov[14] = 1e6            # z (no observado)
        cov[21] = 1e6            # roll
        cov[28] = 1e6            # pitch
        cov[35] = yaw_var        # yaw
        m.pose.covariance = cov
        self._pub_est.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
