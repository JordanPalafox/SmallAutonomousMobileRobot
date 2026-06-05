#!/usr/bin/env python3
"""Clasificador de logos Amazon / Pepsi / Walmart con YOLO.

Carga un modelo Ultralytics (weights.pt) entrenado con tres clases
{0: amazon, 1: pepsi, 2: walmart} y ordena las detecciones de izquierda
a derecha (posicion 1, 2, 3 segun la coordenada X del centro del bbox).

Salidas
-------
    /logo_order   std_msgs/String  JSON  -> {"order": ["pepsi", "amazon", "walmart"]}
    /logo_debug   sensor_msgs/Image      -> frame anotado (rqt_image_view)

Uso
---
    ros2 run perception logo_classifier
    ros2 run perception logo_classifier --ros-args -p model_path:=/ruta/weights.pt
    ros2 run perception logo_classifier --ros-args -p show_window:=false
    ros2 run perception logo_classifier --ros-args -p qos:=reliable

Teclas (ventana): q / ESC = salir.
"""
from __future__ import annotations

import json
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
from std_msgs.msg import String

WIN = 'logo_classifier'

CLASS_COLORS = {
    'amazon':  (0, 165, 255),
    'pepsi':   (0, 0, 200),
    'walmart': (200, 150, 0),
}
DEFAULT_COLOR = (180, 180, 180)
WHITE  = (240, 240, 240)
BLACK  = (20,  20,  20)
GREEN  = (60,  200, 60)
YELLOW = (0,   220, 220)
RED    = (40,  40,  230)


def _text(img: np.ndarray, txt: str, org: tuple, color=WHITE,
          scale: float = 0.55) -> None:
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 4, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class LogoClassifier(Node):
    def __init__(self) -> None:
        super().__init__('logo_classifier')

        self.declare_parameter('model_path', 'src/perception/config/weights.pt')
        self.declare_parameter('image_topic', '/video_source/raw')
        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('process_hz', 15.0)
        self.declare_parameter('show_window', True)
        self.declare_parameter('publish_debug', True)
        # 'best_effort' o 'reliable'
        self.declare_parameter('qos', 'best_effort')

        model_path      = str(self.get_parameter('model_path').value)
        self._topic     = str(self.get_parameter('image_topic').value)
        self._conf      = float(self.get_parameter('conf_threshold').value)
        hz              = float(self.get_parameter('process_hz').value)
        self._show      = bool(self.get_parameter('show_window').value)
        self._pub_dbg_en = bool(self.get_parameter('publish_debug').value)
        qos_name        = str(self.get_parameter('qos').value).lower()

        # ---- Modelo ----
        model_path = self._resolve_model(model_path)
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._class_names: dict[int, str] = self._model.names
            self.get_logger().info(
                f'Modelo cargado: {model_path}  clases={self._class_names}')
        except Exception as exc:
            self.get_logger().error(f'No pude cargar el modelo: {exc!r}')
            raise

        # ---- QoS ----
        if qos_name == 'reliable':
            img_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                durability=DurabilityPolicy.VOLATILE,
            )
        else:
            img_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                durability=DurabilityPolicy.VOLATILE,
            )

        # ---- ROS I/O ----
        self._bridge = CvBridge()
        self.create_subscription(Image, self._topic, self._image_cb, img_qos)
        self._pub_order = self.create_publisher(String, '/logo_order', 10)
        self._pub_dbg   = self.create_publisher(Image, '/logo_debug', 10)

        # ---- Estado ----
        self._frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._tick_count  = 0
        self._log_every   = max(1, int(hz))  # log cada ~1 s

        if self._show:
            try:
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN, 800, 500)
                # Placeholder inmediato para que se vea la ventana antes del primer frame
                ph = self._placeholder('Esperando imagen...\ntopic: ' + self._topic)
                cv2.imshow(WIN, ph)
                cv2.waitKey(1)
            except Exception as exc:
                self.get_logger().warn(f'Sin GUI ({exc!r}); modo headless.')
                self._show = False

        self.create_timer(1.0 / max(1.0, hz), self._tick)
        self.get_logger().info(
            f'logo_classifier listo  topic={self._topic}  '
            f'qos={qos_name}  conf>={self._conf}  gui={self._show}')

    # ------------------------------------------------------------------
    def _resolve_model(self, path: str) -> str:
        import os
        if os.path.isfile(path):
            return path
        try:
            from ament_index_python.packages import get_package_share_directory
            candidate = os.path.join(
                get_package_share_directory('perception'), 'config', 'weights.pt')
            if os.path.isfile(candidate):
                self.get_logger().info(f'Modelo desde share: {candidate}')
                return candidate
        except Exception:
            pass
        return path

    # ------------------------------------------------------------------
    def _image_cb(self, msg: Image) -> None:
        frame = None

        # Intento 1: cv_bridge
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(
                f'cv_bridge fallo ({exc!r}), convirtiendo manualmente...',
                throttle_duration_sec=5.0)

        # Intento 2: conversion manual
        if frame is None:
            try:
                data = np.frombuffer(msg.data, dtype=np.uint8)
                frame = data.reshape((msg.height, msg.width, -1))
                enc = (msg.encoding or '').lower()
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif enc.startswith('rgb'):
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as exc:
                self.get_logger().error(
                    f'No pude decodificar imagen (enc={msg.encoding}): {exc!r}',
                    throttle_duration_sec=5.0)
                return

        self._frame_count += 1
        if self._frame_count == 1:
            self.get_logger().info(
                f'Primer frame recibido: {msg.width}x{msg.height} '
                f'enc={msg.encoding}  shape={frame.shape}')

        self._frame = np.ascontiguousarray(frame)

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        self._tick_count += 1

        # Heartbeat cada ~1 s
        if self._tick_count % self._log_every == 0:
            self.get_logger().info(
                f'[mon] frames_recibidos={self._frame_count}  '
                f'frame_actual={"OK" if self._frame is not None else "None"}',
                throttle_duration_sec=1.0)

        # ---- Si aun no hay frame: mostrar placeholder y salir ----
        if self._frame is None:
            if self._show:
                try:
                    ph = self._placeholder(
                        f'Esperando imagen en {self._topic} ...')
                    cv2.imshow(WIN, ph)
                    cv2.waitKey(1)
                except Exception:
                    pass
            return

        frame = self._frame

        # ---- Inferencia ----
        try:
            results = self._model(frame, conf=self._conf, verbose=False)[0]
        except Exception as exc:
            self.get_logger().warn(f'Inferencia fallo: {exc!r}',
                                   throttle_duration_sec=3.0)
            if self._show:
                try:
                    cv2.imshow(WIN, frame)
                    cv2.waitKey(1)
                except Exception:
                    pass
            return

        detections: list[dict] = []
        for box in results.boxes:
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            name = self._class_names.get(cls_id, str(cls_id))
            detections.append({
                'name': name,
                'conf': conf_val,
                'cx':   cx,
                'box':  (int(x1), int(y1), int(x2), int(y2)),
            })

        # Ordenar izquierda → derecha
        detections.sort(key=lambda d: d['cx'])

        # ---- Publicar orden ----
        self._pub_order.publish(
            String(data=json.dumps({'order': [d['name'] for d in detections]})))

        # ---- Dibujar ----
        vis = self._draw(frame, detections)

        if self._pub_dbg_en:
            try:
                self._pub_dbg.publish(
                    self._bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
            except Exception:
                pass

        if not self._show:
            return
        try:
            cv2.imshow(WIN, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                rclpy.shutdown()
        except Exception as exc:
            self.get_logger().warn(f'GUI fallo ({exc!r}); headless.')
            self._show = False

    # ------------------------------------------------------------------
    def _placeholder(self, msg: str) -> np.ndarray:
        ph = np.zeros((360, 640, 3), dtype=np.uint8)
        for i, line in enumerate(msg.split('\n')):
            _text(ph, line, (20, 60 + i * 36), YELLOW, scale=0.7)
        _text(ph, 'logo_classifier', (20, 340), WHITE, scale=0.5)
        return ph

    # ------------------------------------------------------------------
    def _draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        for rank, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = det['box']
            name  = det['name']
            conf  = det['conf']
            color = CLASS_COLORS.get(name, DEFAULT_COLOR)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f'{name}  {conf:.0%}'
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = y1 - 6 if y1 > lh + 10 else y2 + lh + 6
            cv2.rectangle(out, (x1, ly - lh - 4), (x1 + lw + 4, ly + 2), color, -1)
            cv2.putText(out, label, (x1 + 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

            # Circulo con numero de orden
            cx_i = (x1 + x2) // 2
            cy_i = max(y1, 20)
            cv2.circle(out, (cx_i, cy_i), 18, color, -1)
            cv2.circle(out, (cx_i, cy_i), 18, WHITE, 2)
            rank_txt = str(rank)
            (rtw, rth), _ = cv2.getTextSize(
                rank_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(out, rank_txt,
                        (cx_i - rtw // 2, cy_i + rth // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2, cv2.LINE_AA)

        # Panel inferior
        panel_h = 38
        cv2.rectangle(out, (0, h - panel_h), (w, h), (30, 30, 30), -1)
        if detections:
            order_str = '  ->  '.join(
                f'{i+1}:{d["name"]}' for i, d in enumerate(detections))
            color_st = GREEN
        else:
            order_str = 'Sin detecciones'
            color_st = YELLOW
        _text(out, order_str, (8, h - 10), color_st, scale=0.65)

        _text(out, f'frames:{self._frame_count}  det:{len(detections)}',
              (8, 22), WHITE, scale=0.45)

        return out


def main() -> None:
    rclpy.init()
    node = LogoClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
