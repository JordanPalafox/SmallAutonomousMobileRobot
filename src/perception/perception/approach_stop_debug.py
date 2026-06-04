#!/usr/bin/env python3
"""Detector de "punto de paro" para el avance final del PICK (debug manual).

Problema que resuelve
---------------------
Hoy el estado ``Pick`` avanza contra el roller/pallet con ``drive_until_stall``:
empuja hasta que las ruedas se frenan por el choque. Ese stall mecanico es lo
que dispara el pico de corriente que tira la alimentacion de la Jetson en la
powerbank. Queremos parar *por vision* ANTES de tocar la carga.

Este nodo NO mueve el robot. Es una herramienta de calibracion: tu mueves el
robot a mano hacia la carga y observas dos metodos que estiman "ya estoy
suficientemente cerca, hay que parar":

  1) Ventana "brillo_proximidad":
        Promedio de pixel en una ROI central. Cuando la carga llena la camara,
        el promedio se dispara (mas cerca -> mas brillo) o se desploma (carga
        oscura). Trackbar ``invertir`` cubre ambos casos.

  2) Ventana "contornos":
        Densidad de bordes (Canny) y numero de contornos en la ROI. Cuando la
        carga llena el cuadro de forma uniforme, los bordes desaparecen: "si ya
        no detecta contornos -> punto ideal".

Cada ventana muestra su propio veredicto y el veredicto COMBINADO (AND/OR/solo
uno) con anti-rebote (``hold frames``). El resultado se publica como
``/approach_stop/should_stop`` (std_msgs/Bool) para engancharlo despues al
estado Pick, y como imagen anotada en ``/approach_stop/debug_image``.

Uso
---
    ros2 run perception approach_stop_debug
    # o apuntando a otro topico / sin ventanas:
    ros2 run perception approach_stop_debug --ros-args \
        -p image_topic:=/video_source/raw -p show_window:=true

Teclas en las ventanas:  q o ESC = salir.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

WIN_BRIGHT = 'brillo_proximidad'
WIN_CONT = 'contornos'

# Colores BGR
RED = (40, 40, 230)
GREEN = (60, 200, 60)
CYAN = (220, 220, 0)
WHITE = (240, 240, 240)
BLACK = (20, 20, 20)


def _noop(_v: int) -> None:
    """Callback obligatorio para cv2.createTrackbar (no hacemos nada)."""
    return None


def _odd(n: int) -> int:
    """Devuelve un kernel impar >= 1 (GaussianBlur exige impar)."""
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def _roi_box(w: int, h: int, pct: int) -> tuple[int, int, int, int]:
    """Rectangulo central que cubre ``pct``% del ancho/alto. (x0,y0,x1,y1)."""
    pct = max(5, min(100, int(pct))) / 100.0
    rw, rh = int(w * pct), int(h * pct)
    x0 = (w - rw) // 2
    y0 = (h - rh) // 2
    return x0, y0, x0 + rw, y0 + rh


class ApproachStopDebug(Node):
    def __init__(self) -> None:
        super().__init__('approach_stop_debug')

        # ---- Parametros ----
        self.declare_parameter('image_topic', '/video_source/raw')
        self.declare_parameter('qos', 'sensor_data')  # sensor_data | reliable
        self.declare_parameter('show_window', True)
        self.declare_parameter('display_scale', 2.0)  # upscale para ver mejor
        self.declare_parameter('process_hz', 30.0)
        self.declare_parameter('publish_topic', '/approach_stop/should_stop')

        self._image_topic = str(self.get_parameter('image_topic').value)
        self._qos_name = str(self.get_parameter('qos').value).lower()
        self._show = bool(self.get_parameter('show_window').value)
        self._scale = float(self.get_parameter('display_scale').value)
        hz = float(self.get_parameter('process_hz').value)
        pub_topic = str(self.get_parameter('publish_topic').value)

        # ---- QoS imagen ----
        if self._qos_name in ('reliable', 'default'):
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

        # ---- ROS I/O ----
        self._bridge = CvBridge()
        self.create_subscription(Image, self._image_topic, self._image_cb, img_qos)
        self._pub_stop = self.create_publisher(Bool, pub_topic, 10)
        self._pub_dbg = self.create_publisher(Image, '/approach_stop/debug_image', 10)

        # ---- Estado ----
        self._frame: Optional[np.ndarray] = None
        self._hold_count = 0
        self._last_stop = False

        # ---- Ventanas + trackbars ----
        if self._show:
            try:
                self._build_windows()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f'No se pudo abrir GUI ({exc!r}); corriendo headless. '
                    f'Igual se publica {pub_topic}.'
                )
                self._show = False

        self.create_timer(1.0 / max(1.0, hz), self._tick)
        self.get_logger().info(
            f'approach_stop_debug listo  topic={self._image_topic}  '
            f'qos={self._qos_name}  gui={self._show}  publica={pub_topic}'
        )

    # ------------------------------------------------------------------
    def _build_windows(self) -> None:
        cv2.namedWindow(WIN_BRIGHT, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_CONT, cv2.WINDOW_NORMAL)

        # Ventana brillo (incluye los controles globales de decision)
        cv2.createTrackbar('ROI %', WIN_BRIGHT, 60, 100, _noop)
        cv2.createTrackbar('brillo thr', WIN_BRIGHT, 150, 255, _noop)
        cv2.createTrackbar('invertir(1=cerca oscuro)', WIN_BRIGHT, 0, 1, _noop)
        cv2.createTrackbar('modo 0AND 1OR 2bri 3cont', WIN_BRIGHT, 1, 3, _noop)
        cv2.createTrackbar('hold frames', WIN_BRIGHT, 5, 30, _noop)

        # Ventana contornos
        cv2.createTrackbar('canny lo', WIN_CONT, 50, 500, _noop)
        cv2.createTrackbar('canny hi', WIN_CONT, 150, 500, _noop)
        cv2.createTrackbar('blur', WIN_CONT, 5, 15, _noop)
        cv2.createTrackbar('min area px', WIN_CONT, 100, 3000, _noop)
        cv2.createTrackbar('edge dens x1000', WIN_CONT, 25, 200, _noop)

    def _tb(self, name: str, win: str, default: int) -> int:
        """Lee un trackbar; si la GUI esta apagada usa el default."""
        if not self._show:
            return default
        try:
            return int(cv2.getTrackbarPos(name, win))
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------
    def _image_cb(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:  # noqa: BLE001
            # Fallback manual (mismo enfoque robusto que el dashboard).
            try:
                data = np.frombuffer(msg.data, dtype=np.uint8)
                frame = data.reshape((msg.height, msg.width, -1))
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif (msg.encoding or '').lower().startswith('rgb'):
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'No pude decodificar imagen: {exc!r}')
                return
        self._frame = np.ascontiguousarray(frame)

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        frame = self._frame
        if frame is None:
            return

        h, w = frame.shape[:2]

        # ---- Lee controles ----
        roi_pct = self._tb('ROI %', WIN_BRIGHT, 60)
        bri_thr = self._tb('brillo thr', WIN_BRIGHT, 150)
        invert = self._tb('invertir(1=cerca oscuro)', WIN_BRIGHT, 0)
        mode = self._tb('modo 0AND 1OR 2bri 3cont', WIN_BRIGHT, 1)
        hold_n = self._tb('hold frames', WIN_BRIGHT, 5)
        canny_lo = self._tb('canny lo', WIN_CONT, 50)
        canny_hi = self._tb('canny hi', WIN_CONT, 150)
        blur_k = _odd(self._tb('blur', WIN_CONT, 5))
        min_area = self._tb('min area px', WIN_CONT, 100)
        edge_thr = self._tb('edge dens x1000', WIN_CONT, 25) / 1000.0

        x0, y0, x1, y1 = _roi_box(w, h, roi_pct)
        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ===== Metodo 1: brillo / color promedio =====
        mean_gray = float(gray.mean())
        mean_bgr = roi.reshape(-1, 3).mean(axis=0)  # (B, G, R)
        if invert:
            bright_stop = mean_gray < bri_thr   # carga oscura: cerca = mas oscuro
        else:
            bright_stop = mean_gray > bri_thr   # carga clara: cerca = mas brillo

        # ===== Metodo 2: contornos / densidad de bordes =====
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges = cv2.Canny(blurred, canny_lo, canny_hi)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        sig = [c for c in contours if cv2.contourArea(c) >= min_area]
        n_sig = len(sig)
        roi_area = float(max(1, roi.shape[0] * roi.shape[1]))
        largest_ratio = (
            max(cv2.contourArea(c) for c in sig) / roi_area if sig else 0.0
        )
        # "Ya no hay contornos" = cuadro uniforme (carga pegada y lisa).
        cont_stop = (edge_density < edge_thr) or (n_sig == 0)

        # ===== Decision combinada + anti-rebote =====
        if mode == 0:
            raw_stop = bright_stop and cont_stop
        elif mode == 1:
            raw_stop = bright_stop or cont_stop
        elif mode == 2:
            raw_stop = bright_stop
        else:
            raw_stop = cont_stop

        if raw_stop:
            self._hold_count = min(self._hold_count + 1, 9999)
        else:
            self._hold_count = 0
        should_stop = self._hold_count >= max(1, hold_n)

        self._pub_stop.publish(Bool(data=bool(should_stop)))
        if should_stop != self._last_stop:
            self.get_logger().info(
                f'should_stop -> {should_stop}  '
                f'(brillo={mean_gray:.0f}/{bri_thr} stop={bright_stop} | '
                f'edge_dens={edge_density:.3f}/{edge_thr:.3f} '
                f'cont={n_sig} stop={cont_stop})'
            )
            self._last_stop = should_stop

        # ===== Render =====
        if not self._show:
            return

        info = {
            'roi': (x0, y0, x1, y1),
            'mean_gray': mean_gray, 'mean_bgr': mean_bgr, 'bri_thr': bri_thr,
            'bright_stop': bright_stop,
            'edges': edges, 'edge_density': edge_density, 'edge_thr': edge_thr,
            'sig': sig, 'n_sig': n_sig, 'largest_ratio': largest_ratio,
            'cont_stop': cont_stop, 'should_stop': should_stop,
            'hold': self._hold_count, 'hold_n': hold_n,
        }
        try:
            bright_view = self._draw_bright(frame, info)
            cont_view = self._draw_cont(frame, info)
            cv2.imshow(WIN_BRIGHT, self._scaled(bright_view))
            cv2.imshow(WIN_CONT, self._scaled(cont_view))
            # Publica la vista de brillo anotada para el dashboard.
            try:
                self._pub_dbg.publish(
                    self._bridge.cv2_to_imgmsg(bright_view, encoding='bgr8')
                )
            except Exception:  # noqa: BLE001
                pass
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                rclpy.shutdown()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'GUI fallo ({exc!r}); paso a headless.')
            self._show = False

    # ------------------------------------------------------------------
    def _scaled(self, img: np.ndarray) -> np.ndarray:
        if self._scale and self._scale != 1.0:
            return cv2.resize(
                img, None, fx=self._scale, fy=self._scale,
                interpolation=cv2.INTER_NEAREST,
            )
        return img

    def _draw_bright(self, frame: np.ndarray, info: dict) -> np.ndarray:
        out = frame.copy()
        x0, y0, x1, y1 = info['roi']
        cv2.rectangle(out, (x0, y0), (x1, y1), CYAN, 1)
        b, g, r = (int(v) for v in info['mean_bgr'])
        cv2.rectangle(out, (4, 4), (24, 24), (b, g, r), -1)
        cv2.rectangle(out, (4, 4), (24, 24), WHITE, 1)
        _text(out, f'brillo {info["mean_gray"]:.0f}/{info["bri_thr"]}', (30, 18))
        _text(out, f'BGR {b},{g},{r}', (30, 34))
        _tag(out, 'BRILLO', info['bright_stop'], (out.shape[1] - 92, 12))
        self._banner(out, info)
        return out

    def _draw_cont(self, frame: np.ndarray, info: dict) -> np.ndarray:
        out = frame.copy()
        x0, y0, x1, y1 = info['roi']
        # Pinta los bordes Canny (rojo) dentro de la ROI.
        edges = info['edges']
        mask = edges > 0
        roi_view = out[y0:y1, x0:x1]
        roi_view[mask] = RED
        # Dibuja los contornos significativos (verde). offset reubica las
        # coordenadas (que vienen relativas a la ROI) sobre el frame completo.
        if info['sig']:
            cv2.drawContours(out, info['sig'], -1, GREEN, 1, offset=(x0, y0))
        cv2.rectangle(out, (x0, y0), (x1, y1), CYAN, 1)
        _text(out, f'edge_dens {info["edge_density"]:.3f}/{info["edge_thr"]:.3f}',
              (6, 18))
        _text(out, f'contornos {info["n_sig"]}  max {info["largest_ratio"]:.2f}',
              (6, 34))
        _tag(out, 'CONT', info['cont_stop'], (out.shape[1] - 92, 12))
        self._banner(out, info)
        return out

    def _banner(self, out: np.ndarray, info: dict) -> None:
        h, w = out.shape[:2]
        stop = info['should_stop']
        color = RED if stop else GREEN
        msg = 'ALTO - DETENER' if stop else 'AVANZAR'
        cv2.rectangle(out, (0, h - 24), (w, h), color, -1)
        _text(out, f'{msg}  [{info["hold"]}/{info["hold_n"]}]',
              (6, h - 7), BLACK)


def _text(img, txt, org, color=WHITE) -> None:
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLACK, 3,
                cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                cv2.LINE_AA)


def _tag(img, name, stop, org) -> None:
    color = RED if stop else GREEN
    _text(img, f'{name}:{"STOP" if stop else "ok"}', org, color)


def main() -> None:
    rclpy.init()
    node = ApproachStopDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
