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
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseArray, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Int32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

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


def _quat_from_R(R: np.ndarray) -> tuple[float, float, float, float]:
    """Cuaternion (x, y, z, w) desde una matriz de rotacion 3x3."""
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _load_track(path: str) -> Optional[tuple[float, float]]:
    """Lee `track: {x, y}` del aruco_map.yaml (extension de la pista canonica)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        t = data.get('track') if isinstance(data, dict) else None
        if isinstance(t, dict) and 'x' in t and 'y' in t:
            return float(t['x']), float(t['y'])
    except Exception:
        pass
    return None


def _load_camera_yaml(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    cam = data['camera']
    K = np.array(cam['camera_matrix'], dtype=np.float64).reshape(3, 3)
    dist = np.array(cam['distortion_coefficients'], dtype=np.float64).flatten()
    return K, dist


def _marker_rotation(yaw: float, mount: str) -> np.ndarray:
    """Rotacion R_map_marker segun el montaje.

    Convencion ArUco del marker: X derecha, Y arriba, Z saliendo de la cara
    hacia el observador.

    mount='wall' (VERTICAL, en muro/costado de obstaculo): la cara mira en
        horizontal. `yaw` = direccion de la NORMAL del marker (hacia donde
        apunta su cara) en el plano XY del mapa.
            Z_marker = (cos yaw, sin yaw, 0)   (normal, horizontal)
            Y_marker = (0, 0, 1)               (arriba del mundo)
            X_marker = Y x Z = (-sin yaw, cos yaw, 0)
    mount='floor' (PLANO en el piso): Z_marker = +Z del mapa (arriba),
        `yaw` = giro alrededor de Z.
    """
    if mount == 'floor':
        return _rot_z(yaw)
    cf, sf = math.cos(yaw), math.sin(yaw)
    return np.array([[-sf, 0.0, cf],
                     [ cf, 0.0, sf],
                     [0.0, 1.0, 0.0]], dtype=np.float64)


def _load_aruco_map(path: str, inplane_deg: float = 0.0) -> Dict[int, np.ndarray]:
    """Lee aruco_map.yaml -> {id: T_map_marker (4x4)}.

    Campos por marker:
        id, x, y         posicion del centro del marker [m]
        z                altura del centro sobre el piso [m] (clave en muro)
        yaw_deg          direccion a la que MIRA el marker (normal) [grados]
                         (alias aceptados: theta_deg en grados, theta en rad)
        mount            'wall' (vertical, default) | 'floor' (plano)
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    markers = data.get('markers', data) if isinstance(data, dict) else data
    # Acepta lista [{id, x, y, ...}, ...] o dict {id: {x, y, ...}, ...}.
    if isinstance(markers, dict):
        items = [dict(m, id=m.get('id', k)) for k, m in markers.items()]
    else:
        items = list(markers)
    out: Dict[int, np.ndarray] = {}
    for m in items:
        mid = int(m['id'])
        x = float(m['x'])
        y = float(m['y'])
        z = float(m.get('z', 0.10))
        if 'yaw_deg' in m:
            yaw = math.radians(float(m['yaw_deg']))
        elif 'theta_deg' in m:
            yaw = math.radians(float(m['theta_deg']))
        else:
            yaw = float(m.get('theta', 0.0))
        mount = str(m.get('mount', 'wall')).lower()
        # Rotación in-plane EXTRA alrededor del NORMAL del marker. Corrige la
        # convención con que el sim (o la realidad) pega la IMAGEN del ArUco: si
        # la "imagen-arriba" no apunta al cielo (p.ej. gz rota la textura 90°),
        # el detector da un frame girado y la pose sale mal/saltando por marker.
        # Ajustar marker_inplane_deg (0/90/180/270) hasta que la pose cuadre.
        R = _marker_rotation(yaw, mount) @ _rot_z(math.radians(inplane_deg))
        out[mid] = _make_tf(R, [x, y, z])
    return out


class ArucoLocalizationNode(Node):
    def __init__(self) -> None:
        super().__init__('aruco_localization')

        # ---- I/O ----
        self.declare_parameter('image_topic', '/cam_img')
        self.declare_parameter('qos', 'sensor_data')
        self.declare_parameter('camera_params', 'src/perception/config/camera_params.yaml')
        self.declare_parameter('aruco_map', 'src/perception/config/aruco_map.yaml')
        # Giro in-plane de la convención de la imagen del marker (0/90/180/270).
        self.declare_parameter('marker_inplane_deg', 0.0)
        self.declare_parameter('marker_length', 0.09)
        self.declare_parameter('dictionary', 'original')
        self.declare_parameter('map_frame', 'map')
        # Pose del ORIGEN del aruco_map.yaml dentro del frame `map` del SLAM
        # [x, y, yaw_deg]. Sirve cuando mediste los ArUco desde una esquina
        # pero el SLAM tiene su origen en otro punto (p.ej. el robot arranca en
        # el CENTRO de la pista -> origen = -(ancho/2, alto/2)).
        self.declare_parameter('aruco_origin_in_map', [0.0, 0.0, 0.0])

        # ---- Extrinsecos camara (frame OPTICO en base_link) ----
        # cam_xyz: origen del frame optico de la camara en base_link [m].
        # cam_pitch_deg: inclinacion hacia ABAJO de la camara [grados].
        #   Markers en MURO (verticales) -> camara casi horizontal (0).
        #   AJUSTAR al montaje real.
        self.declare_parameter('cam_xyz', [0.10, 0.0, 0.16])  # camara frontal, 16 cm del piso
        self.declare_parameter('cam_pitch_deg', 0.0)
        self.declare_parameter('cam_yaw_deg', 0.0)   # desalineacion lateral, normalmente 0

        # ---- Filtrado / fusion ----
        self.declare_parameter('max_range', 2.0)        # m; descarta markers mas lejanos
        self.declare_parameter('min_markers', 1)        # detecciones minimas para publicar
        self.declare_parameter('xy_std_base', 0.03)     # m, ruido base a 1 m
        self.declare_parameter('yaw_std_base_deg', 3.0)  # grados, ruido base a 1 m

        # ---- Tambien publicar /aruco_poses (para no correr 2 nodos) ----
        self.declare_parameter('publish_poses', True)
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('publish_debug_image', True)
        # Overlay RViz del mapa IDEAL (frame canonico): paredes + ArUco en su
        # posicion conocida. Sirve de BLANCO visual: ves cuanto le falta al mapa del
        # SLAM y como debe "saltar" encima al anclar. Se publica latched (transient).
        self.declare_parameter('publish_ideal_map', True)
        # DIAGNOSTICO TEMPORAL del flip de PnP: por marker con ambiguedad imprime las
        # 2 soluciones IPPE (facing, uprightness, reproj, pose del robot resultante) y
        # cual se eligio. Sirve para ver por que la heuristica falla y elegir bien.
        # Apagar con publish_debug:=false ... mejor un flag propio: debug_pnp.
        self.declare_parameter('debug_pnp', True)
        self.declare_parameter('debug_pnp_period', 1.5)   # s entre volcados (no floodear)
        # HÍBRIDO contra el flip de 90°: con >=2 markers se estima la pose del robot por
        # un ajuste rígido 2D de POSICIONES de markers (inmune a la orientación/flip de
        # PnP); con 1 marker se cae al single-marker (floor-constraint). NO exige 2.
        # ArUco = SOLO anclar el mapa (FASE 1, stream /aruco_markers) + rescatar el MCL
        # cuando se pierde (FASE 2, /aruco_pose_estimate). La pose per-frame se publica
        # SOLO cuando es CONFIABLE: ajuste por POSICIONES de ≥2 markers (inmune al flip
        # de PnP). Con <2 markers / fit pobre NO se publica pose (nada de single-marker
        # volteable). Así el rescate nunca re-siembra hacia una pose chueca.
        self.declare_parameter('multi_marker_enabled', True)
        self.declare_parameter('multi_marker_baseline_gate', 0.25)  # m; baseline mínimo de markers para fiarse del fit
        self.declare_parameter('multi_marker_max_rmse', 0.10)        # m; RMS máximo del fit (si no, no publica)

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
        self._publish_ideal = bool(self.get_parameter('publish_ideal_map').value)
        self._debug_pnp = bool(self.get_parameter('debug_pnp').value)
        self._debug_pnp_period = float(self.get_parameter('debug_pnp_period').value)
        self._last_pnp_log = 0.0
        self._multi_enabled = bool(self.get_parameter('multi_marker_enabled').value)
        self._multi_baseline_gate = float(self.get_parameter('multi_marker_baseline_gate').value)
        self._multi_max_rmse = float(self.get_parameter('multi_marker_max_rmse').value)

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
        marker_inplane = float(self.get_parameter('marker_inplane_deg').value)
        self._marker_inplane_rad = math.radians(marker_inplane)  # el horneado en self._map
        self._map = _load_aruco_map(map_path, marker_inplane)
        self._track = _load_track(map_path)   # (tx, ty) de la pista canonica o None

        # Reubica el mapa ArUco al frame `map` del SLAM (si el origen difiere).
        origin = [float(v) for v in self.get_parameter('aruco_origin_in_map').value]
        o_x = origin[0] if len(origin) > 0 else 0.0
        o_y = origin[1] if len(origin) > 1 else 0.0
        o_yaw = math.radians(origin[2]) if len(origin) > 2 else 0.0
        if abs(o_x) > 1e-9 or abs(o_y) > 1e-9 or abs(o_yaw) > 1e-9:
            T_pre = _make_tf(_rot_z(o_yaw), [o_x, o_y, 0.0])
            self._map = {mid: T_pre @ T for mid, T in self._map.items()}
            self.get_logger().info(
                f'aruco_map reubicado: origen en map=({o_x:.3f}, {o_y:.3f}, '
                f'{math.degrees(o_yaw):.1f}deg)')

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
        # Marcadores RViz: linea ArUco -> pose estimada del robot (debug visual).
        self._pub_markers = self.create_publisher(
            MarkerArray, '/aruco_localization/debug_markers', 10)

        # Stream por-marker para el ANCLAJE multi-marker del SLAM. Float64MultiArray
        # plano: [sec, nanosec, (id, base_x, base_y, canon_x, canon_y) x N]. Por cada
        # marker MAPEADO y en rango manda su posicion en base_link + su posicion
        # canonica (del aruco_map). El SLAM acumula ids distintos, los sube a su frame
        # de mapa y ajusta Umeyama -> ancla el mapa con una esquina en (0,0).
        self._pub_anchor = self.create_publisher(Float64MultiArray, '/aruco_markers', 10)

        # Overlay del mapa IDEAL (latched para late-joiners + timer 1 Hz por si RViz
        # arranca tarde). Frame canonico (`map`): paredes + ArUco en su sitio conocido.
        if self._publish_ideal:
            ideal_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST, depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._pub_ideal = self.create_publisher(
                MarkerArray, '/aruco_ideal_map', ideal_qos)
            self._ideal_msg = self._build_ideal_map()
            self._pub_ideal.publish(self._ideal_msg)
            self.create_timer(1.0, self._publish_ideal_map)
            self.get_logger().info(
                f'Overlay mapa ideal en /aruco_ideal_map ({len(self._map)} ArUco'
                + (f', pista {self._track[0]:.2f}x{self._track[1]:.2f} m' if self._track else '')
                + ', frame map)')

        self.get_logger().info(f'aruco_localization listo en {image_topic}')

    # ------------------------------------------------------------------
    def _robot_pose_from_marker(self, det: dict, log: bool = False):
        """Pose (x, y, yaw, dist) del robot en el mapa desde UN SOLO marker, o None.

        Un marker plano tiene ambiguedad de orientacion (flip IPPE) y ademas la
        convencion in-plane de la imagen puede estar mal (el diagnostico mostro el
        robot 'tumbado' ~90deg). Pero el robot va SOBRE EL PISO (derecho, solo yaw),
        y la orientacion del marker es CONOCIDA -> esa restriccion resuelve todo:
        se prueba cada flip x cada inplane {0,90,180,270} y se queda con la
        combinacion que deja al robot DERECHO (su eje Z apunta a +Z del mundo,
        upright~+1). De ahi sale el yaw correcto; la posicion = p_map - Rz(yaw)@p_base
        con la traslacion medida (tvec). Asi UN marker basta para fijar (x,y,yaw),
        inmune al flip y a la convencion in-plane. Verificado exacto numericamente.
        """
        T_map_marker = self._map.get(int(det['id']))
        if T_map_marker is None:
            return None
        tvec = np.asarray(det['tvec'], dtype=np.float64).flatten()
        dist = float(np.linalg.norm(tvec))
        if dist > self._max_range or dist <= 1e-3:
            return None
        R_base_cam = self._T_base_cam[:3, :3]
        t_base_cam = self._T_base_cam[:3, 3]
        # Orientacion FISICA base del marker (sin el inplane horneado en self._map).
        R_marker_base = T_map_marker[:3, :3] @ _rot_z(-self._marker_inplane_rad)
        p_map = T_map_marker[:3, 3]

        # Candidatos: ambas soluciones IPPE (si las hay) x 4 convenciones in-plane.
        sols = det.get('solutions') or [{'rvec': np.asarray(det['rvec']).flatten()}]
        best = None   # (upright, yaw, inplane_deg, sol_idx)
        for si, sol in enumerate(sols):
            R_cm, _ = cv2.Rodrigues(np.asarray(sol['rvec'], dtype=np.float64).reshape(3, 1))
            for ip_deg in (0.0, 90.0, 180.0, 270.0):
                R_mm = R_marker_base @ _rot_z(math.radians(ip_deg))
                R_map_base = R_mm @ R_cm.T @ R_base_cam.T
                upright = float(R_map_base[2, 2])   # +1 = robot derecho (piso)
                if best is None or upright > best[0]:
                    yaw = math.atan2(R_map_base[1, 0], R_map_base[0, 0])
                    best = (upright, yaw, ip_deg, si)

        upright, yaw, ip_deg, si = best
        p_base = R_base_cam @ tvec + t_base_cam
        xy = p_map[:2] - (_rot_z(yaw) @ p_base)[:2]
        if log:
            self.get_logger().info(
                f'[1mk id={int(det["id"])}] yaw={math.degrees(yaw):+.0f}deg '
                f'upright={upright:+.2f} inplane={ip_deg:.0f} sol={si} '
                f'pos=({xy[0]:.2f},{xy[1]:.2f}) dist={dist:.2f}m')
        return float(xy[0]), float(xy[1]), float(yaw), dist

    @staticmethod
    def _fit_robot_pose_2d(src_list, dst_list):
        """Ajuste rígido 2D (Umeyama, sin escala): halla (x,y,yaw) tal que
        dst ≈ Rz(yaw)·src + (x,y). src = posiciones de markers en base_link,
        dst = sus posiciones canónicas (map). La transformación ES la pose del
        robot (el origen de base_link es (0,0)). INMUNE al flip/inplane de PnP
        (solo usa posiciones). Devuelve (x, y, yaw, rmse, baseline, residuos)."""
        src = np.asarray(src_list, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(dst_list, dtype=np.float64).reshape(-1, 2)
        sc = src.mean(axis=0); dc = dst.mean(axis=0)
        S = (dst - dc).T @ (src - sc)
        yaw = math.atan2(S[1, 0] - S[0, 1], S[0, 0] + S[1, 1])
        cc, ss = math.cos(yaw), math.sin(yaw)
        R = np.array([[cc, -ss], [ss, cc]])
        t = dc - R @ sc
        pred = (R @ src.T).T + t
        res = np.sqrt(np.sum((pred - dst) ** 2, axis=1))
        rmse = float(np.sqrt(np.mean(res ** 2)))
        n = len(src)
        baseline = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                baseline = max(baseline, float(np.linalg.norm(src[i] - src[j])))
        return float(t[0]), float(t[1]), float(yaw), rmse, baseline, res

    @classmethod
    def _fit_robot_pose_2d_robust(cls, src_list, dst_list, inlier_tol):
        """Como _fit_robot_pose_2d pero con rechazo greedy de 1+ outliers
        (drop-worst): descarta el marker con mayor residuo mientras supere
        inlier_tol y queden >=2. Devuelve (x, y, yaw, rmse, baseline, n)."""
        src = list(src_list); dst = list(dst_list)
        while len(src) >= 2:
            x, y, yaw, rmse, baseline, res = cls._fit_robot_pose_2d(src, dst)
            k = int(np.argmax(res))
            if res[k] <= inlier_tol or len(src) <= 2:
                return x, y, yaw, rmse, baseline, len(src)
            del src[k]; del dst[k]
        return None

    def _image_cb(self, msg: Image) -> None:
        frame = imgmsg_to_bgr(self._bridge, msg)
        if frame is None:
            return
        dets = self._detector.detect(frame)

        if self._pub_poses is not None:
            self._publish_raw_poses(dets, msg)

        # Stream por-marker para el anclaje multi-marker (independiente del fix
        # fusionado: el SLAM acumula ids distintos en el tiempo, basta 1 por frame).
        self._publish_marker_pairs(dets, msg)

        # Log gateado por tiempo (no floodear).
        now_s = self.get_clock().now().nanoseconds * 1e-9
        do_log = self._debug_pnp and (now_s - self._last_pnp_log) >= self._debug_pnp_period
        if do_log:
            self._last_pnp_log = now_s

        # ---- Pose del robot: MULTI (≥2, inmune al flip) primario, SINGLE-marker respaldo ----
        # Como en SIM (donde con 1 aruco funciona muy bien): con ≥2 markers se ajusta por
        # POSICIONES (inmune al flip de PnP); con 1 marker se usa el single-marker con
        # restricción de piso (_robot_pose_from_marker, auto-inplane). El single-marker
        # depende de los extrínsecos de cámara correctos (cam_pitch/cam_xyz) — por eso en
        # el real hay que ponerlos como en el gemelo/URDF (pitch ~20°).
        base_xy, canon_xy, dists, used_markers, sm_dets = [], [], [], [], []
        for det in dets:
            mid = int(det['id'])
            T = self._map.get(mid)
            if T is None:
                continue
            tvec = np.asarray(det['tvec'], dtype=np.float64).flatten()
            dist = float(np.linalg.norm(tvec))
            if dist > self._max_range or dist <= 1e-3:
                continue
            p_base = (self._T_base_cam @ np.array([tvec[0], tvec[1], tvec[2], 1.0]))[:2]
            base_xy.append(np.asarray(p_base, float)); canon_xy.append(T[:2, 3]); dists.append(dist)
            mpos = T[:3, 3]
            used_markers.append((mid, float(mpos[0]), float(mpos[1]), float(mpos[2])))
            sm_dets.append(det)
        used = len(used_markers)

        if self._pub_debug is not None:
            annotated = self._detector.draw_detections(frame, dets)
            out = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out.header.stamp = msg.header.stamp
            out.header.frame_id = self._camera_frame
            self._pub_debug.publish(out)

        if used < self._min_markers:
            return

        x_f = y_f = yaw_f = xy_var = yaw_var = None
        tag = ''
        # PRIMARIO: ≥2 markers → ajuste por posiciones (inmune al flip).
        if self._multi_enabled and used >= 2:
            fit = self._fit_robot_pose_2d_robust(base_xy, canon_xy, self._multi_max_rmse)
            if fit is not None:
                fx, fy, fyaw, rmse, baseline, n = fit
                if n >= 2 and baseline >= self._multi_baseline_gate and rmse <= self._multi_max_rmse:
                    x_f, y_f, yaw_f = fx, fy, fyaw
                    xy_var = max(rmse * rmse, 1e-4)
                    yaw_var = max((rmse / baseline) ** 2, math.radians(1.0) ** 2)
                    tag = f'multi N={n}/{used} baseline={baseline:.2f} rmse={rmse:.3f}'
        # RESPALDO: 1 marker (o multi no fiable) → single-marker (como en sim).
        if x_f is None:
            sm_xs, sm_ys, sm_yaws, sm_w = [], [], [], []
            for det, dist in zip(sm_dets, dists):
                est = self._robot_pose_from_marker(det, log=do_log)
                if est is None:
                    continue
                x, y, yaw, _ = est
                sm_xs.append(x); sm_ys.append(y); sm_yaws.append(yaw); sm_w.append(1.0 / (dist * dist))
            if not sm_w:
                return
            w = np.asarray(sm_w); wsum = float(w.sum()); nsm = len(sm_w)
            x_f = float(np.dot(w, sm_xs) / wsum)
            y_f = float(np.dot(w, sm_ys) / wsum)
            yaw_f = math.atan2(float(np.dot(w, np.sin(sm_yaws))), float(np.dot(w, np.cos(sm_yaws))))
            dist_eff = math.sqrt(1.0 / (wsum / nsm))
            xy_var = (self._xy_std_base * dist_eff) ** 2 / nsm
            yaw_var = (self._yaw_std_base * dist_eff) ** 2 / nsm
            tag = f'single N={nsm}'

        self._publish_estimate(x_f, y_f, yaw_f, xy_var, yaw_var, msg.header.stamp,
                               multi=tag.startswith('multi'))
        self._publish_debug_markers(used_markers, x_f, y_f, msg.header.stamp)
        if do_log:
            self.get_logger().info(
                f'ArUco [{tag}] pose=({x_f:.2f},{y_f:.2f},{math.degrees(yaw_f):.0f}deg) '
                f'ids={[m[0] for m in used_markers]}')

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

    def _publish_marker_pairs(self, dets, msg: Image) -> None:
        """Publica /aruco_markers: por cada marker MAPEADO y en rango, su posicion
        en base_link + su posicion canonica (frame fijo del aruco_map). El SLAM lo
        usa para el anclaje rigido multi-marker (Umeyama)."""
        out = [float(msg.header.stamp.sec), float(msg.header.stamp.nanosec)]
        for det in dets:
            mid = int(det['id'])
            T_map_marker = self._map.get(mid)
            if T_map_marker is None:
                continue
            tv = np.asarray(det['tvec'], dtype=np.float64).flatten()
            dist = float(np.linalg.norm(tv))
            if dist > self._max_range or dist <= 1e-3:
                continue
            base = (self._T_base_cam @ np.array([tv[0], tv[1], tv[2], 1.0]))[:3]
            canon = T_map_marker[:3, 3]
            out += [float(mid), float(base[0]), float(base[1]),
                    float(canon[0]), float(canon[1])]
        if len(out) <= 2:   # ningun marker mapeado en rango este frame
            return
        m = Float64MultiArray()
        m.data = out
        self._pub_anchor.publish(m)

    # ------------------------------------------------------------------
    def _build_ideal_map(self) -> MarkerArray:
        """MarkerArray del mapa IDEAL en el frame canonico (`map`): paredes
        exteriores + cada ArUco (placa orientada + flecha de la normal + ID).
        Es el BLANCO al que el SLAM debe anclar (esquina SW en (0,0))."""
        ma = MarkerArray()
        frame = self._map_frame

        # Lead with DELETEALL so RViz drops markers it cached from a PREVIOUS run
        # (e.g. ArUcos removed from the map) instead of leaving them drawn forever.
        # rviz2 processes the array atomically before rendering → the ADDs below
        # repaint the current set with no visible flicker. Included in every publish
        # so a still-running RViz self-cleans whenever this node restarts.
        _clear = Marker()
        _clear.action = Marker.DELETEALL
        ma.markers.append(_clear)

        def base(ns: str, mid: int, mtype: int) -> Marker:
            mk = Marker()
            mk.header.frame_id = frame
            mk.ns = ns
            mk.id = int(mid)
            mk.type = mtype
            mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0
            return mk

        # 1. Paredes exteriores + origen (0,0)
        if self._track is not None:
            tx, ty = self._track
            wall = base('ideal_walls', 0, Marker.LINE_STRIP)
            wall.scale.x = 0.03
            wall.color.r, wall.color.g, wall.color.b, wall.color.a = 0.1, 0.9, 0.3, 0.9
            for cx, cy in [(0, 0), (tx, 0), (tx, ty), (0, ty), (0, 0)]:
                p = Point(); p.x = float(cx); p.y = float(cy); p.z = 0.0
                wall.points.append(p)
            ma.markers.append(wall)

            org = base('ideal_origin', 0, Marker.SPHERE)
            org.scale.x = org.scale.y = org.scale.z = 0.12
            org.color.r, org.color.g, org.color.b, org.color.a = 1.0, 1.0, 0.0, 1.0
            ma.markers.append(org)
            otxt = base('ideal_origin', 1, Marker.TEXT_VIEW_FACING)
            otxt.pose.position.y = -0.18
            otxt.scale.z = 0.12
            otxt.color.r = otxt.color.g = otxt.color.b = otxt.color.a = 1.0
            otxt.text = '(0,0)'
            ma.markers.append(otxt)

        # 2. ArUco canonicos (placa + normal + ID)
        for mid, T in sorted(self._map.items()):
            pos = T[:3, 3]
            R = T[:3, :3]
            qx, qy, qz, qw = _quat_from_R(R)

            cube = base('ideal_aruco', mid, Marker.CUBE)
            cube.pose.position.x = float(pos[0])
            cube.pose.position.y = float(pos[1])
            cube.pose.position.z = float(pos[2])
            cube.pose.orientation.x = qx
            cube.pose.orientation.y = qy
            cube.pose.orientation.z = qz
            cube.pose.orientation.w = qw
            cube.scale.x = 0.09; cube.scale.y = 0.09; cube.scale.z = 0.012  # placa fina (cara ⟂ normal)
            cube.color.r, cube.color.g, cube.color.b, cube.color.a = 0.1, 0.5, 1.0, 0.95
            ma.markers.append(cube)

            n = R[:, 2]   # normal del marker (hacia donde MIRA la cara)
            arr = base('ideal_aruco_normal', mid, Marker.ARROW)
            s = Point(); s.x = float(pos[0]); s.y = float(pos[1]); s.z = float(pos[2])
            e = Point()
            e.x = float(pos[0] + 0.25 * n[0])
            e.y = float(pos[1] + 0.25 * n[1])
            e.z = float(pos[2] + 0.25 * n[2])
            arr.points = [s, e]
            arr.scale.x = 0.015; arr.scale.y = 0.035; arr.scale.z = 0.04
            arr.color.r, arr.color.g, arr.color.b, arr.color.a = 1.0, 0.2, 0.2, 0.9
            ma.markers.append(arr)

            txt = base('ideal_aruco_id', mid, Marker.TEXT_VIEW_FACING)
            txt.pose.position.x = float(pos[0])
            txt.pose.position.y = float(pos[1])
            txt.pose.position.z = float(pos[2] + 0.16)
            txt.scale.z = 0.11
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = str(mid)
            ma.markers.append(txt)

        return ma

    def _publish_ideal_map(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for mk in self._ideal_msg.markers:
            mk.header.stamp = stamp
        self._pub_ideal.publish(self._ideal_msg)

    def _publish_estimate(self, x, y, yaw, xy_var, yaw_var, stamp, multi=False) -> None:
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
        # BANDERA fuente del fix en el término cruzado x-y (no lo usa nadie como
        # covarianza real; slam_node onAruco lo lee): 1.0 = MULTI-marker (ajuste por
        # posiciones, INMUNE al flip de PnP/yaw mal mapeado), 0.0 = single-marker
        # (puede salir reflejado si un marker está mal medido). El rescate por
        # discrepancia del SLAM solo se fía de los multi para no teletransportar un
        # MCL bien localizado a partir de un single reflejado.
        cov[1] = 1.0 if multi else 0.0
        m.pose.covariance = cov
        self._pub_est.publish(m)

    def _publish_debug_markers(self, used_markers, rx, ry, stamp) -> None:
        """RViz (frame `map`): una LINEA de cada ArUco detectado a la pose
        estimada del robot, una esfera ROJA en la pose estimada y esferas
        AZULES en cada ArUco usado. Si la localizacion es buena, la esfera
        roja cae sobre el robot real y las lineas salen de los markers que se
        ven en Gazebo. Topic: /aruco_localization/debug_markers."""
        life = Duration(sec=1, nanosec=0)

        links = Marker()
        links.header.frame_id = self._map_frame
        links.header.stamp = stamp
        links.ns = 'aruco_links'
        links.id = 0
        links.type = Marker.LINE_LIST
        links.action = Marker.ADD
        links.scale.x = 0.02
        links.color.g = 1.0
        links.color.a = 1.0
        links.lifetime = life
        links.pose.orientation.w = 1.0

        mk = Marker()
        mk.header.frame_id = self._map_frame
        mk.header.stamp = stamp
        mk.ns = 'aruco_markers'
        mk.id = 1
        mk.type = Marker.SPHERE_LIST
        mk.action = Marker.ADD
        mk.scale.x = mk.scale.y = mk.scale.z = 0.10
        mk.color.b = 1.0
        mk.color.g = 0.4
        mk.color.a = 1.0
        mk.lifetime = life
        mk.pose.orientation.w = 1.0

        for (_mid, mx, my, mz) in used_markers:
            links.points.append(Point(x=mx, y=my, z=mz))
            links.points.append(Point(x=rx, y=ry, z=0.0))
            mk.points.append(Point(x=mx, y=my, z=mz))

        est = Marker()
        est.header.frame_id = self._map_frame
        est.header.stamp = stamp
        est.ns = 'aruco_robot_est'
        est.id = 2
        est.type = Marker.SPHERE
        est.action = Marker.ADD
        est.pose.position.x = rx
        est.pose.position.y = ry
        est.pose.position.z = 0.05
        est.pose.orientation.w = 1.0
        est.scale.x = est.scale.y = est.scale.z = 0.12
        est.color.r = 1.0
        est.color.a = 1.0
        est.lifetime = life

        self._pub_markers.publish(MarkerArray(markers=[links, mk, est]))


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
