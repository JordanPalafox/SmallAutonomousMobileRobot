"""Detector de pose 6DOF para marcadores ArUco planos.

Detecta los marcadores con `cv2.aruco` (compatible con OpenCV 4.6 y 4.7+),
refina las esquinas con `cornerSubPix` y estima la pose con `cv2.solvePnP`
usando el modelo de marker cuadrado plano (`SOLVEPNP_IPPE_SQUARE`).

La interfaz devuelta es identica a la de `QRPoseDetector` para que ambos
detectores sean intercambiables en el resto del paquete; la unica diferencia
es que aqui `id` es un `int` (el ID del marker ArUco) en lugar de texto.

Diccionario por defecto: ARUCO original (rejilla 5x5, 1024 IDs). Es el
diccionario que pide `marker_mapper` y el que detecta mejor la camara del
Puzzlebot (celdas grandes); un 4x4 no es compatible con marker_mapper.

Convencion del marker (igual a OpenCV ArUco):
    origen en el centro del marker, X a la derecha, Y hacia arriba,
    Z saliendo del plano del marker hacia la camara. Por lo tanto
    `tvec.z` es la distancia camara->marker (positiva).
"""
from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np
from geometry_msgs.msg import Pose


# ---------------------------------------------------------------------------
# Quaternion helper (mismo que qr_pose_detector para evitar dependencias cruzadas)
# ---------------------------------------------------------------------------
def _rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _rvec_tvec_to_pose(rvec: np.ndarray, tvec: np.ndarray) -> Pose:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    qx, qy, qz, qw = _rotation_matrix_to_quaternion(R)
    pose = Pose()
    t = tvec.flatten()
    pose.position.x = float(t[0])
    pose.position.y = float(t[1])
    pose.position.z = float(t[2])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


# Nombres de diccionario aceptados -> constante de cv2.aruco.
# 'original' = DICT_ARUCO_ORIGINAL (rejilla 5x5, 1024 markers).
_DICT_ALIASES = {
    'original': 'DICT_ARUCO_ORIGINAL',
    'aruco_original': 'DICT_ARUCO_ORIGINAL',
    '5x5': 'DICT_5X5_1000',
    '5x5_1000': 'DICT_5X5_1000',
    '4x4': 'DICT_4X4_1000',
    '6x6': 'DICT_6X6_1000',
    '7x7': 'DICT_7X7_1000',
}


def _resolve_dictionary(name: str):
    """Devuelve el objeto diccionario de cv2.aruco para un nombre dado."""
    key = name.strip().lower()
    const_name = _DICT_ALIASES.get(key)
    if const_name is None:
        # Permitir pasar el nombre completo de la constante, p.ej. 'DICT_ARUCO_ORIGINAL'.
        const_name = name if name.startswith('DICT_') else 'DICT_ARUCO_ORIGINAL'
    const = getattr(cv2.aruco, const_name)
    # OpenCV 4.7+ renombro getPredefinedDictionary -> mismo nombre, API estable.
    return cv2.aruco.getPredefinedDictionary(const), const_name


class ArucoDetector:
    """Detecta marcadores ArUco y estima su pose 6DOF.

    Args:
        camera_matrix: matriz intrinseca 3x3.
        dist_coeffs:   coeficientes de distorsion (5+ elementos).
        marker_length: lado fisico del marker en metros (9 cm = 0.09).
        dictionary:    nombre del diccionario ('original' = ARUCO 5x5).
        refine:        si refinar las esquinas con cornerSubPix.
    """

    # Orden de esquinas de cv2.aruco.detectMarkers: TL, TR, BR, BL.
    # En el frame del marker (origen al centro, X derecha, Y arriba, Z saliendo)
    # los puntos 3D correspondientes son:
    _CORNER_TEMPLATE = np.array(
        [
            [-0.5,  0.5, 0.0],  # TL
            [ 0.5,  0.5, 0.0],  # TR
            [ 0.5, -0.5, 0.0],  # BR
            [-0.5, -0.5, 0.0],  # BL
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        marker_length: float = 0.09,
        dictionary: str = 'original',
        refine: bool = True,
    ) -> None:
        self._K = np.array(camera_matrix, dtype=np.float64).reshape(3, 3)
        self._dist = np.array(dist_coeffs, dtype=np.float64).flatten()
        self._L = float(marker_length)
        self._refine = bool(refine)
        self._obj_pts = (self._CORNER_TEMPLATE * self._L).astype(np.float64)

        self._dict, self._dict_name = _resolve_dictionary(dictionary)

        # API nueva (OpenCV >= 4.7): cv2.aruco.ArucoDetector + DetectorParameters().
        # API vieja (OpenCV <= 4.6, p.ej. 4.5.4 del Jetson): detectMarkers(img,
        # dict, parameters=...) y DetectorParameters_create().
        if hasattr(cv2.aruco, 'DetectorParameters'):
            self._params = cv2.aruco.DetectorParameters()
        else:  # OpenCV <= 4.6
            self._params = cv2.aruco.DetectorParameters_create()
        self._new_api = hasattr(cv2.aruco, 'ArucoDetector')
        if self._new_api:
            self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)
        else:
            self._detector = None

        # IPPE_SQUARE: solver optimo para markers planos cuadrados (mucho mas
        # estable que ITERATIVE). Disponible en OpenCV 4.0+.
        self._pnp_flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)

    @property
    def dictionary(self) -> str:
        return self._dict_name

    @property
    def marker_length(self) -> float:
        return self._L

    # ------------------------------------------------------------------
    def _detect_corners(self, gray: np.ndarray):
        """Devuelve (corners_list, ids_list). corners shape (4,2)."""
        if self._new_api:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:  # pragma: no cover
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dict, parameters=self._params,
            )
        if ids is None or len(ids) == 0:
            return [], []
        corners_list = [np.asarray(c, dtype=np.float32).reshape(-1, 2) for c in corners]
        ids_list = [int(i) for i in np.asarray(ids).flatten()]
        return corners_list, ids_list

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[dict]:
        """Detecta markers y devuelve lista de dicts con pose.

        Cada dict contiene:
            id      - int, ID del marker ArUco
            corners - np.ndarray (4,2) TL,TR,BR,BL en pixeles
            rvec    - np.ndarray (3,)
            tvec    - np.ndarray (3,) en metros, Z = distancia
            pose    - geometry_msgs/Pose en frame de la camara
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners_list, ids_list = self._detect_corners(gray)
        if not corners_list:
            return []

        h, w = gray.shape[:2]
        results: list[dict] = []
        for corners, marker_id in zip(corners_list, ids_list):
            corners_f = corners.astype(np.float32).copy()
            if corners_f.shape != (4, 2):
                continue
            if not np.isfinite(corners_f).all():
                continue

            # cornerSubPix exige que el rect de busqueda quepa en la imagen.
            if self._refine:
                win = 5
                pad = win + 2
                xs, ys = corners_f[:, 0], corners_f[:, 1]
                inside = (
                    (xs >= pad).all() and (xs < w - pad).all()
                    and (ys >= pad).all() and (ys < h - pad).all()
                )
                if inside:
                    try:
                        term = (
                            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30, 1e-3,
                        )
                        refined = cv2.cornerSubPix(
                            gray, corners_f.reshape(-1, 1, 2),
                            (win, win), (-1, -1), term,
                        )
                        corners_f = refined.reshape(-1, 2)
                    except cv2.error:
                        pass

            # solvePnPGeneric da las (hasta 2) soluciones IPPE de un marker plano
            # (ambigüedad de pose). Elegimos la FRONTAL: la que tiene el normal
            # del marker apuntando HACIA la cámara (dot(normal, tvec) < 0). Así la
            # pose del ArUco nunca "apunta hacia atrás", solo hacia el frente.
            try:
                n_sol, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                    self._obj_pts, corners_f.astype(np.float64),
                    self._K, self._dist, flags=self._pnp_flag)
            except cv2.error:
                continue
            if not n_sol:
                continue
            best = None
            solutions = []   # TODAS las soluciones IPPE (para desambiguar con el mapa aguas abajo)
            for i in range(int(n_sol)):
                rv = np.asarray(rvecs[i]).reshape(3, 1)
                tv = np.asarray(tvecs[i]).reshape(3)
                Rm, _ = cv2.Rodrigues(rv)
                facing = float(np.dot(Rm[:, 2], tv))   # <0 → normal hacia la cámara (frontal)
                err = float(np.asarray(reproj[i]).reshape(-1)[0]) if reproj is not None else 0.0
                solutions.append({'rvec': rv.flatten(), 'tvec': tv.reshape(3).copy(),
                                  'reproj': err, 'facing': facing})
                score = (0 if facing < 0.0 else 1, err)  # prioriza frontal; luego menor error
                if best is None or score < best[0]:
                    best = (score, rv, tv)
            rvec = best[1]; tvec = best[2].reshape(3, 1)

            results.append({
                'id': int(marker_id),
                'corners': corners_f,
                'rvec': rvec.flatten(),
                'tvec': tvec.flatten(),
                'pose': _rvec_tvec_to_pose(rvec, tvec),
                # Ambigüedad planar: ambas soluciones IPPE. El localizador elige la
                # correcta con la orientación CONOCIDA del marker (robot derecho),
                # no con la heurística de "frontal" (falla por ratos a distancia).
                'solutions': solutions,
            })

        return results

    # ------------------------------------------------------------------
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Dibuja contorno, ejes 3D y el ID sobre una copia del frame."""
        out = frame.copy()
        for det in detections:
            corners = det['corners'].astype(np.int32)
            cv2.polylines(out, [corners], True, (0, 255, 0), 2)
            for idx, (x, y) in enumerate(det['corners']):
                color = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)][idx]
                cv2.circle(out, (int(x), int(y)), 4, color, -1)

            try:
                cv2.drawFrameAxes(
                    out, self._K, self._dist,
                    det['rvec'], det['tvec'], self._L * 0.5, 2,
                )
            except cv2.error:
                pass

            cx = int(det['corners'][:, 0].mean())
            cy = int(det['corners'][:, 1].mean())
            cv2.putText(out, f"id={det['id']}", (cx - 20, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out


def euler_from_rvec(rvec: np.ndarray) -> tuple[float, float, float]:
    """Convierte rvec a (roll, pitch, yaw) en radianes, convencion XYZ."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)
