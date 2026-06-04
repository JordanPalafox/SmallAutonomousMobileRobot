#!/usr/bin/env python3
"""Detector de "punto de paro" por el logo Electric 80 (debug manual).

Por que este enfoque
--------------------
El pallet ya NO tapa la camara, asi que brillo/contornos (ver
approach_stop_debug.py) dejaron de servir. En cambio el logo "Electric 80"
(≡80) impreso en la carga es un objeto de tamano conocido: su tamano aparente
en pixeles crece de forma monotona conforme el robot se acerca. Eso lo vuelve
un proxy de distancia muy confiable y facil de calibrar.

Metodo primario (validado): TEMPLATE MATCHING MULTIESCALA.
    Se guarda una plantilla del logo TAL COMO SE VE EN EL PUNTO IDEAL de paro.
    Cada frame se busca esa plantilla redimensionada a varias escalas; la
    escala con mejor correlacion = tamano aparente actual relativo al ideal:
        escala < 1.0  -> el logo se ve mas chico  -> todavia LEJOS, avanzar.
        escala ~ 1.0  -> el logo ya tiene el tamano del punto ideal -> ALTO.
    (En pruebas: ideal->1.00, simulando lejos->0.85/0.70/0.55, score>0.99.)
    Robusto al desenfoque (es correlacion) y, restringido a una ROI inferior,
    ignora el poster del rack de fondo.

Metodo secundario (cross-check): ORB + homografia.
    Empareja features del logo y proyecta su caja; proxy = ancho de la caja en
    px. Maneja perspectiva/rotacion pero sufre con el desenfoque de cerca.

Salidas
-------
    /approach_stop/should_stop  std_msgs/Bool   -> engancha al estado Pick.
    /approach_stop/center_error geometry_msgs/Point  -> x = error de centrado
                                                        del logo en px (alineacion).
    /approach_stop/debug_image  sensor_msgs/Image     -> vista anotada (dashboard).

Uso
---
    ros2 run perception logo_stop_debug
    # re-calibrar la referencia EN el robot: pon el robot en el punto ideal,
    # encuadra el logo en la franja inferior y presiona 's' en la ventana.

Teclas:  s = guardar referencia desde el frame actual | q / ESC = salir.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

WIN_T = 'logo_template'
WIN_O = 'logo_orb'

RED = (40, 40, 230)
GREEN = (60, 200, 60)
CYAN = (220, 220, 0)
YELLOW = (0, 220, 220)
WHITE = (240, 240, 240)
BLACK = (20, 20, 20)

# Caja (proporciones del frame) usada para recortar la referencia con 's'.
# Coincide con el recorte del logo en la posicion ideal (320x240).
REF_BOX = (0.625, 0.99, 0.125, 0.94)  # (y0, y1, x0, x1)


def _noop(_v: int) -> None:
    return None


class LogoStopDebug(Node):
    def __init__(self) -> None:
        super().__init__('logo_stop_debug')

        # ---- Parametros ----
        self.declare_parameter('image_topic', '/video_source/raw')
        self.declare_parameter('qos', 'sensor_data')
        self.declare_parameter(
            'template_path', 'src/perception/config/e80_logo_ref.png')
        self.declare_parameter('show_window', True)
        self.declare_parameter('display_scale', 2.0)
        self.declare_parameter('process_hz', 15.0)
        self.declare_parameter('scale_min', 0.30)
        self.declare_parameter('scale_max', 1.20)
        self.declare_parameter('scale_step', 0.05)
        # Umbrales de decision. En modo GUI los trackbars mandan; headless
        # (Jetson) usan estos como default -> configurables por --ros-args.
        self.declare_parameter('roi_top_pct', 30)
        self.declare_parameter('match_thr', 0.45)
        self.declare_parameter('stop_scale', 0.98)
        self.declare_parameter('mode', 2)        # 0AND 1OR 2tmpl 3orb
        self.declare_parameter('hold_frames', 4)
        self.declare_parameter('min_inliers', 12)
        self.declare_parameter('stop_width', 210)
        # Publicar la vista anotada aunque corra headless (Jetson) para verla
        # en remoto con rqt_image_view. Apagar en produccion para ahorrar CPU.
        self.declare_parameter('publish_debug', True)

        self._image_topic = str(self.get_parameter('image_topic').value)
        qos_name = str(self.get_parameter('qos').value).lower()
        self._show = bool(self.get_parameter('show_window').value)
        self._scale = float(self.get_parameter('display_scale').value)
        hz = float(self.get_parameter('process_hz').value)
        self._tpl_path = str(self.get_parameter('template_path').value)
        smin = float(self.get_parameter('scale_min').value)
        smax = float(self.get_parameter('scale_max').value)
        sstep = float(self.get_parameter('scale_step').value)
        n = max(2, int(round((smax - smin) / sstep)) + 1)
        self._scales = [round(smin + sstep * i, 3) for i in range(n)]

        # Defaults de decision (sirven como valor inicial de trackbars y como
        # valor fijo cuando se corre headless en la Jetson).
        self._d_roi = int(self.get_parameter('roi_top_pct').value)
        self._d_match = int(round(float(self.get_parameter('match_thr').value) * 100))
        self._d_stopscale = int(round(float(self.get_parameter('stop_scale').value) * 100))
        self._d_mode = int(self.get_parameter('mode').value)
        self._d_hold = int(self.get_parameter('hold_frames').value)
        self._d_inl = int(self.get_parameter('min_inliers').value)
        self._d_width = int(self.get_parameter('stop_width').value)
        self._publish_debug = bool(self.get_parameter('publish_debug').value)

        # ---- QoS ----
        if qos_name in ('reliable', 'default'):
            img_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST, depth=10,
                durability=DurabilityPolicy.VOLATILE)
        else:
            img_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, depth=5,
                durability=DurabilityPolicy.VOLATILE)

        # ---- ROS I/O ----
        self._bridge = CvBridge()
        self.create_subscription(Image, self._image_topic, self._image_cb, img_qos)
        self._pub_stop = self.create_publisher(Bool, '/approach_stop/should_stop', 10)
        self._pub_center = self.create_publisher(Point, '/approach_stop/center_error', 10)
        self._pub_dbg = self.create_publisher(Image, '/approach_stop/debug_image', 10)

        # ---- Estado ----
        self._frame: Optional[np.ndarray] = None
        self._hold = 0
        self._last_stop = False
        self._tick_i = 0
        self._log_every = max(1, int(round(hz)))  # ~1 log/seg en headless
        self._orb = cv2.ORB_create(nfeatures=600)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ref_gray: Optional[np.ndarray] = None
        self._ref_kp = None
        self._ref_des = None
        self._tpl_path = self._resolve_template_path(self._tpl_path)
        self._load_reference(self._tpl_path)

        if self._show:
            try:
                self._build_windows()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'Sin GUI ({exc!r}); headless.')
                self._show = False

        self.create_timer(1.0 / max(1.0, hz), self._tick)
        self.get_logger().info(
            f'logo_stop_debug listo  topic={self._image_topic}  '
            f'ref={"OK" if self._ref_gray is not None else "FALTA (presiona s)"}  '
            f'escalas={self._scales[0]}..{self._scales[-1]}  gui={self._show}')

    # ------------------------------------------------------------------
    def _resolve_template_path(self, path: str) -> str:
        """Si el path dado no existe (p.ej. CWD distinto en la Jetson), cae al
        e80_logo_ref.png instalado en el share del paquete."""
        if os.path.isfile(path):
            return path
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('perception')
            cand = os.path.join(share, 'config', 'e80_logo_ref.png')
            if os.path.isfile(cand):
                self.get_logger().info(f'Referencia desde share: {cand}')
                return cand
        except Exception:  # noqa: BLE001
            pass
        return path

    def _load_reference(self, path: str) -> bool:
        ref = cv2.imread(path, cv2.IMREAD_COLOR)
        if ref is None:
            self.get_logger().warn(f'No pude leer referencia en {path!r}.')
            return False
        self._ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        self._ref_kp, self._ref_des = self._orb.detectAndCompute(self._ref_gray, None)
        self.get_logger().info(
            f'Referencia {ref.shape[1]}x{ref.shape[0]} '
            f'orb_kp={0 if self._ref_kp is None else len(self._ref_kp)}')
        return True

    def _build_windows(self) -> None:
        cv2.namedWindow(WIN_T, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_O, cv2.WINDOW_NORMAL)
        # Controles globales + del metodo template en la ventana WIN_T.
        cv2.createTrackbar('ROI top %', WIN_T, self._d_roi, 90, _noop)
        cv2.createTrackbar('match thr x100', WIN_T, self._d_match, 100, _noop)
        cv2.createTrackbar('stop escala x100', WIN_T, self._d_stopscale, 130, _noop)
        cv2.createTrackbar('modo 0AND 1OR 2tmpl 3orb', WIN_T, self._d_mode, 3, _noop)
        cv2.createTrackbar('hold frames', WIN_T, self._d_hold, 30, _noop)
        # Controles del metodo ORB en WIN_O.
        cv2.createTrackbar('min inliers', WIN_O, self._d_inl, 60, _noop)
        cv2.createTrackbar('stop ancho px', WIN_O, self._d_width, 320, _noop)

    def _tb(self, name: str, win: str, default: int) -> int:
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
                self.get_logger().warn(f'No decodifique imagen: {exc!r}')
                return
        self._frame = np.ascontiguousarray(frame)

    # ------------------------------------------------------------------
    def _match_template_multiscale(self, scene_gray):
        """Devuelve dict con score, escala, caja (x,y,w,h) o None."""
        if self._ref_gray is None:
            return None
        Hs, Ws = scene_gray.shape
        rh, rw = self._ref_gray.shape
        best = None
        for s in self._scales:
            tw, th = int(rw * s), int(rh * s)
            if tw < 12 or th < 12 or tw > Ws or th > Hs:
                continue
            t = cv2.resize(self._ref_gray, (tw, th))
            res = cv2.matchTemplate(scene_gray, t, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if best is None or mv > best['score']:
                best = {'score': float(mv), 'scale': float(s),
                        'box': (ml[0], ml[1], tw, th)}
        return best

    def _match_orb(self, scene_gray):
        """Devuelve dict con inliers, poligono (4x2), ancho, centro o None."""
        if self._ref_des is None or len(self._ref_kp) < 4:
            return None
        kp, des = self._orb.detectAndCompute(scene_gray, None)
        if des is None or len(kp) < 4:
            return None
        try:
            knn = self._bf.knnMatch(self._ref_des, des, k=2)
        except cv2.error:
            return None
        good = [m for m, n in (p for p in knn if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return {'inliers': 0, 'good': len(good), 'poly': None,
                    'width': 0.0, 'cx': None}
        src = np.float32([self._ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hmat, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if Hmat is None:
            return {'inliers': 0, 'good': len(good), 'poly': None,
                    'width': 0.0, 'cx': None}
        inliers = int(mask.sum()) if mask is not None else 0
        rh, rw = self._ref_gray.shape
        corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, Hmat).reshape(-1, 2)
        width = float(proj[:, 0].max() - proj[:, 0].min())
        cx = float(proj[:, 0].mean())
        return {'inliers': inliers, 'good': len(good), 'poly': proj,
                'width': width, 'cx': cx}

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        frame = self._frame
        if frame is None:
            return
        h, w = frame.shape[:2]

        roi_top_pct = self._tb('ROI top %', WIN_T, self._d_roi)
        match_thr = self._tb('match thr x100', WIN_T, self._d_match) / 100.0
        stop_scale = self._tb('stop escala x100', WIN_T, self._d_stopscale) / 100.0
        mode = self._tb('modo 0AND 1OR 2tmpl 3orb', WIN_T, self._d_mode)
        hold_n = self._tb('hold frames', WIN_T, self._d_hold)
        min_inliers = self._tb('min inliers', WIN_O, self._d_inl)
        stop_width = self._tb('stop ancho px', WIN_O, self._d_width)

        ry0 = int(h * max(0, min(90, roi_top_pct)) / 100.0)
        roi = frame[ry0:h, 0:w]
        scene_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ---- Template multiescala ----
        tm = self._match_template_multiscale(scene_gray)
        tmpl_ok = tm is not None and tm['score'] >= match_thr
        tmpl_stop = tmpl_ok and (tm['scale'] >= stop_scale)
        tmpl_cx = None
        if tm is not None:
            x, y, bw, bh = tm['box']
            tmpl_cx = x + bw / 2.0  # en coords de la ROI (x global == ROI x)

        # ---- ORB homografia ----
        om = self._match_orb(scene_gray)
        orb_ok = om is not None and om['inliers'] >= min_inliers and om['poly'] is not None
        orb_stop = orb_ok and (om['width'] >= stop_width)

        # ---- Decision combinada + anti-rebote ----
        if mode == 0:
            raw = tmpl_stop and orb_stop
        elif mode == 1:
            raw = tmpl_stop or orb_stop
        elif mode == 2:
            raw = tmpl_stop
        else:
            raw = orb_stop
        self._hold = self._hold + 1 if raw else 0
        should_stop = self._hold >= max(1, hold_n)

        self._pub_stop.publish(Bool(data=bool(should_stop)))
        # Error de centrado (px) del metodo activo, para alineacion futura.
        cx = tmpl_cx if mode != 3 else (om['cx'] if orb_ok else None)
        if cx is not None:
            self._pub_center.publish(Point(x=float(cx - w / 2.0), y=0.0, z=0.0))
        if should_stop != self._last_stop:
            sc = f'{tm["scale"]:.2f}' if tm else 'NA'
            scr = f'{tm["score"]:.2f}' if tm else 'NA'
            o_inl = om['inliers'] if om else 0
            o_w = f'{om["width"]:.0f}' if om else 'NA'
            self.get_logger().info(
                f'should_stop -> {should_stop}  '
                f'tmpl(score={scr} escala={sc} stop={tmpl_stop}) '
                f'orb(inl={o_inl} w={o_w} stop={orb_stop})')
            self._last_stop = should_stop

        # Monitor periodico (util en headless mientras mueves el robot a mano).
        self._tick_i += 1
        if self._tick_i % self._log_every == 0:
            sc = f'{tm["scale"]:.2f}' if tm else 'NA'
            scr = f'{tm["score"]:.2f}' if tm else 'NA'
            self.get_logger().info(
                f'[mon] tmpl score={scr} escala={sc}/{stop_scale:.2f}  '
                f'should_stop={should_stop}')

        # Vista anotada (template). Se publica siempre que publish_debug este
        # activo -> visible en remoto con rqt_image_view aunque corra headless.
        tview = None
        if self._publish_debug or self._show:
            tview = self._draw_template(frame, ry0, tm, tmpl_ok, tmpl_stop,
                                        match_thr, stop_scale, should_stop,
                                        hold_n)
            if self._publish_debug:
                try:
                    self._pub_dbg.publish(
                        self._bridge.cv2_to_imgmsg(tview, encoding='bgr8'))
                except Exception:  # noqa: BLE001
                    pass

        if not self._show:
            return
        try:
            oview = self._draw_orb(frame, ry0, om, orb_ok, orb_stop,
                                   min_inliers, stop_width, should_stop, hold_n)
            cv2.imshow(WIN_T, self._scaled(tview))
            cv2.imshow(WIN_O, self._scaled(oview))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                rclpy.shutdown()
            elif key == ord('s'):
                self._save_reference(frame)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'GUI fallo ({exc!r}); headless.')
            self._show = False

    # ------------------------------------------------------------------
    def _save_reference(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        y0, y1, x0, x1 = REF_BOX
        crop = frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)].copy()
        ok = cv2.imwrite(self._tpl_path, crop)
        self.get_logger().info(
            f'Referencia {"guardada" if ok else "NO se pudo guardar"} en '
            f'{self._tpl_path} ({crop.shape[1]}x{crop.shape[0]})')
        if ok:
            self._load_reference(self._tpl_path)

    def _scaled(self, img):
        if self._scale and self._scale != 1.0:
            return cv2.resize(img, None, fx=self._scale, fy=self._scale,
                              interpolation=cv2.INTER_NEAREST)
        return img

    def _draw_template(self, frame, ry0, tm, ok, stop, thr, stop_scale,
                       should_stop, hold_n):
        out = frame.copy()
        cv2.line(out, (0, ry0), (out.shape[1], ry0), CYAN, 1)
        if tm is not None:
            x, y, bw, bh = tm['box']
            color = GREEN if stop else (YELLOW if ok else RED)
            cv2.rectangle(out, (x, y + ry0), (x + bw, y + bh + ry0), color, 2)
            cv2.line(out, (x + bw // 2, y + ry0),
                     (x + bw // 2, y + bh + ry0), color, 1)
            _text(out, f'score {tm["score"]:.2f}/{thr:.2f}', (6, 16))
            _text(out, f'escala {tm["scale"]:.2f}  paro>={stop_scale:.2f}', (6, 32))
            cx = x + bw / 2.0
            _text(out, f'centro err {cx - out.shape[1]/2:+.0f}px', (6, 48))
        else:
            _text(out, 'SIN REFERENCIA (presiona s)', (6, 20), RED)
        _tag(out, 'TMPL', stop, ok, (out.shape[1] - 96, 14))
        _banner(out, should_stop, self._hold, hold_n)
        return out

    def _draw_orb(self, frame, ry0, om, ok, stop, min_inl, stop_w,
                  should_stop, hold_n):
        out = frame.copy()
        cv2.line(out, (0, ry0), (out.shape[1], ry0), CYAN, 1)
        if om is not None and om['poly'] is not None:
            poly = om['poly'].copy()
            poly[:, 1] += ry0
            color = GREEN if stop else (YELLOW if ok else RED)
            cv2.polylines(out, [poly.astype(np.int32)], True, color, 2)
            _text(out, f'inliers {om["inliers"]}/{min_inl} (good {om["good"]})', (6, 16))
            _text(out, f'ancho {om["width"]:.0f}px  paro>={stop_w}', (6, 32))
            if om['cx'] is not None:
                _text(out, f'centro err {om["cx"]-out.shape[1]/2:+.0f}px', (6, 48))
        else:
            n = 0 if om is None else om.get('good', 0)
            _text(out, f'logo no localizado (good {n})', (6, 20), RED)
        _tag(out, 'ORB', stop, ok, (out.shape[1] - 96, 14))
        _banner(out, should_stop, self._hold, hold_n)
        return out


def _text(img, txt, org, color=WHITE) -> None:
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _tag(img, name, stop, ok, org) -> None:
    color = GREEN if stop else (YELLOW if ok else RED)
    state = 'STOP' if stop else ('ok' if ok else '--')
    _text(img, f'{name}:{state}', org, color)


def _banner(out, should_stop, hold, hold_n) -> None:
    h, w = out.shape[:2]
    color = RED if should_stop else GREEN
    msg = 'ALTO - DETENER' if should_stop else 'AVANZAR'
    cv2.rectangle(out, (0, h - 22), (w, h), color, -1)
    _text(out, f'{msg}  [{hold}/{hold_n}]', (6, h - 6), BLACK)


def main() -> None:
    rclpy.init()
    node = LogoStopDebug()
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
