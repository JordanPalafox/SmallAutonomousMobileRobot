"""Detector de pose 6DOF para QR codes planos.

Detecta los QR con `cv2.QRCodeDetector`, refina las esquinas con
`cornerSubPix` y estima la pose con `cv2.solvePnP` usando el modelo de
marker cuadrado plano. La interfaz devuelta es compatible con la del
`ArucoDetector` para integrarse facilmente con el resto del paquete.

Convencion del marker (igual a OpenCV ArUco):
    origen en el centro del QR, X a la derecha, Y hacia arriba,
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
# Quaternion helper (copiado del aruco_detector para evitar dependencias cruzadas)
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


class QRPoseDetector:
    """Detecta QR codes y estima su pose 6DOF a partir de las 4 esquinas.

    Args:
        camera_matrix: matriz intrinseca 3x3.
        dist_coeffs:   coeficientes de distorsion (5+ elementos).
        marker_length: lado fisico del QR en metros.
        refine:        si refinar las esquinas con cornerSubPix.
    """

    # cv2.QRCodeDetector devuelve esquinas en orden:
    #   [top-left, top-right, bottom-right, bottom-left]
    # Para el frame del marker (origen al centro, X derecha, Y arriba, Z saliendo)
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
        marker_length: float = 0.035,
        refine: bool = True,
        backend: str = 'auto',
    ) -> None:
        self._K = np.array(camera_matrix, dtype=np.float64).reshape(3, 3)
        self._dist = np.array(dist_coeffs, dtype=np.float64).flatten()
        self._L = float(marker_length)
        self._refine = bool(refine)
        self._obj_pts = (self._CORNER_TEMPLATE * self._L).astype(np.float64)

        # Backend de deteccion:
        #   'aruco'  - cv2.QRCodeDetectorAruco (OpenCV 4.7+, robusto con QR
        #              pequenos/baja-resolucion porque usa el detector de
        #              cuadrados de ArUco antes de decodificar).
        #   'classic'- cv2.QRCodeDetector (default historico, mas debil).
        #   'auto'   - usa 'aruco' si esta disponible, si no 'classic'.
        self._backend_name = backend.lower()
        if self._backend_name == 'auto':
            self._backend_name = 'aruco' if hasattr(cv2, 'QRCodeDetectorAruco') else 'classic'

        if self._backend_name == 'aruco' and hasattr(cv2, 'QRCodeDetectorAruco'):
            self._detector = cv2.QRCodeDetectorAruco()
        else:
            self._backend_name = 'classic'
            self._detector = cv2.QRCodeDetector()

        # IPPE_SQUARE es el solver optimo para marcadores planos cuadrados
        # (mucho mas estable que ITERATIVE en este caso). Disponible en OpenCV 4.0+
        self._pnp_flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)

    @property
    def backend(self) -> str:
        return self._backend_name

    # ------------------------------------------------------------------
    def _detect_corners(self, gray: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
        """Devuelve (corners_list, data_list). corners shape (4,2)."""
        corners_list: list[np.ndarray] = []
        data_list: list[str] = []

        # Intentar API multi (preferida)
        try:
            ok, decoded, points, _ = self._detector.detectAndDecodeMulti(gray)
        except cv2.error:
            ok, decoded, points = False, [], None

        if ok and points is not None and len(points) > 0:
            for i, pts in enumerate(points):
                pts4 = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                if pts4.shape[0] != 4:
                    continue
                corners_list.append(pts4)
                data_list.append(str(decoded[i]) if i < len(decoded) else '')
            return corners_list, data_list

        # Fallback: single QR
        try:
            data, points, _ = self._detector.detectAndDecode(gray)
        except cv2.error:
            return [], []

        if points is None:
            return [], []
        pts4 = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if pts4.shape[0] != 4:
            return [], []
        corners_list.append(pts4)
        data_list.append(str(data) if data else '')
        return corners_list, data_list

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[dict]:
        """Detecta QRs y devuelve lista de dicts con pose.

        Cada dict contiene:
            id      - texto decodificado (puede ser '' si no decodifico)
            corners - np.ndarray (4,2) TL,TR,BR,BL en pixeles
            rvec    - np.ndarray (3,)
            tvec    - np.ndarray (3,) en metros, Z = distancia
            pose    - geometry_msgs/Pose en frame de la camara
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners_list, data_list = self._detect_corners(gray)
        if not corners_list:
            return []

        h, w = gray.shape[:2]
        results: list[dict] = []
        for corners, data in zip(corners_list, data_list):
            corners_f = corners.astype(np.float32).copy()

            if corners_f.shape != (4, 2):
                continue
            if not np.isfinite(corners_f).all():
                continue

            # cornerSubPix exige que el rect de busqueda quepa en la imagen.
            # Solo refinar si TODAS las esquinas tienen al menos `pad` pixeles
            # de margen al borde (pad = winSize + zeroZone + 2 por seguridad).
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
                        # Si falla la refinacion, seguimos con las esquinas crudas
                        pass

            ok, rvec, tvec = cv2.solvePnP(
                self._obj_pts,
                corners_f.astype(np.float64),
                self._K,
                self._dist,
                flags=self._pnp_flag,
            )
            if not ok:
                continue

            results.append({
                'id': data,
                'corners': corners_f,
                'rvec': rvec.flatten(),
                'tvec': tvec.flatten(),
                'pose': _rvec_tvec_to_pose(rvec, tvec),
            })

        return results

    # ------------------------------------------------------------------
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Dibuja contorno, ejes 3D y texto sobre una copia del frame."""
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
            label = det['id'] if det['id'] else 'QR'
            cv2.putText(out, label, (cx - 20, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out


def euler_from_rvec(rvec: np.ndarray) -> tuple[float, float, float]:
    """Convierte rvec a (roll, pitch, yaw) en radianes, convencion XYZ.

    Util para mostrar la orientacion del marker al usuario.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
    return float(roll), float(pitch), float(yaw)
