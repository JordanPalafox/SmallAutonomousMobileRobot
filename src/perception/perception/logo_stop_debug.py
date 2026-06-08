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

Metodo alternativo (modo 1): edge matching con Canny.
    Correlacion de bordes sobre la ROI; invariante a cambios monotonicos de
    brillo. Mas robusto ante condiciones extremas (muy oscuro / sobreexpuesto).

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

import sys
import termios
import threading
import tty

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
from std_msgs.msg import Bool, String

WIN_T = 'logo_template'

RED = (40, 40, 230)
GREEN = (60, 200, 60)
CYAN = (220, 220, 0)
YELLOW = (0, 220, 220)
WHITE = (240, 240, 240)
BLACK = (20, 20, 20)

# Caja (proporciones del frame) usada para recortar la referencia con 's'.
# Coincide con el recorte del logo en la posicion ideal (320x240).


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
        self.declare_parameter('roi_top_pct', 10)
        self.declare_parameter('match_thr', 0.49)   # calibrado 2026-06-08 roller
        self.declare_parameter('stop_scale', 0.27)  # calibrado 2026-06-08 roller
        self.declare_parameter('mode', 0)        # 0=tmpl 1=edge
        self.declare_parameter('hold_frames', 5)    # calibrado roller
        # Publicar la vista anotada aunque corra headless (Jetson) para verla
        # en remoto con rqt_image_view. Apagar en produccion para ahorrar CPU.
        self.declare_parameter('publish_debug', True)
        # Per-pick-type LOGO profiles: the DEFAULT params above are the ROLLER
        # profile; a RACK profile (its own template + thresholds) is selected
        # when /robot_state == rack_state. active_states (comma list) gates the
        # node — it only matches + publishes when /robot_state is one of them
        # (skips the heavy matchTemplate otherwise, ahorra CPU). Empty = always.
        self.declare_parameter('active_states', '')
        self.declare_parameter('rack_state', 'PICK_FROM_RACK')
        self.declare_parameter('rack_template_path', '')
        self.declare_parameter('rack_stop_scale', 0.05)   # calibrado 2026-06-08 rack
        self.declare_parameter('rack_roi_top_pct', 79)    # calibrado 2026-06-08 rack
        self.declare_parameter('rack_match_thr', 0.43)    # calibrado 2026-06-08 rack
        self.declare_parameter('rack_hold_frames', 8)     # calibrado rack: 6 continuos
        # ---- Robustez lumínica ----
        # CLAHE normaliza la iluminación local (sombras, sobreexposición) ANTES
        # del template matching. Se aplica a la plantilla al cargarla Y a la ROI
        # en cada frame. TM_CCOEFF_NORMED ya resta la media global; CLAHE corrige
        # además la variación LOCAL de contraste que eso no elimina.
        # clahe_grid: nº de celdas por eje (8 → ~40x30 px por celda en 320x240).
        self.declare_parameter('use_clahe', True)
        self.declare_parameter('clahe_clip', 2.0)
        self.declare_parameter('clahe_grid', 8)
        # Modo 4 (edge matching): Canny sobre la ROI y la plantilla; la correlación
        # de bordes es invariante a cambios monotónicos de brillo.  Más robusto que
        # CLAHE ante condiciones extremas (muy oscuro / muy sobreexpuesto).
        # edge_low/high: umbrales de Canny (relación ~1:3 recomendada).
        self.declare_parameter('edge_low', 30)
        self.declare_parameter('edge_high', 100)

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
        self._publish_debug = bool(self.get_parameter('publish_debug').value)
        self._active_states = {s.strip().upper() for s in
            str(self.get_parameter('active_states').value).split(',') if s.strip()}
        self._rack_state = str(self.get_parameter('rack_state').value).strip().upper()
        self._robot_state = ''
        # Preprocesado lumínico
        self._use_clahe = bool(self.get_parameter('use_clahe').value)
        _clahe_clip = float(self.get_parameter('clahe_clip').value)
        _clahe_grid = int(self.get_parameter('clahe_grid').value)
        self._clahe = cv2.createCLAHE(
            clipLimit=_clahe_clip,
            tileGridSize=(_clahe_grid, _clahe_grid),
        )
        self._edge_low = int(self.get_parameter('edge_low').value)
        self._edge_high = int(self.get_parameter('edge_high').value)

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
        self.create_subscription(String, '/robot_state', self._robot_state_cb, 10)
        # Armado: solo publica should_stop=True cuando está armado.
        # El SM arma justo antes del creep post-dock y desarma al terminar.
        #   ros2 topic pub --once /logo_stop/enable std_msgs/msg/Bool 'data: true'
        self._armed = False
        self._arm_time = 0.0          # monotonic timestamp del último armado
        self._arm_grace_s = 0.0       # se fija al armar: 0s roller, 3s rack
        self.create_subscription(Bool, '/logo_stop/enable', self._enable_cb, 10)
        # Guardar foto de referencia headless:
        #   ros2 topic pub --once /logo_stop/save_ref std_msgs/msg/String 'data: centro'
        #   valores: centro | izq | der
        self.create_subscription(String, '/logo_stop/save_ref', self._save_ref_cb, 10)
        self._pub_stop = self.create_publisher(Bool, '/approach_stop/should_stop', 10)
        self._pub_center = self.create_publisher(Point, '/approach_stop/center_error', 10)
        self._pub_dbg = self.create_publisher(Image, '/approach_stop/debug_image', 10)

        # ---- Estado ----
        self._frame: Optional[np.ndarray] = None
        self._hold = 0
        self._last_stop = False
        self._tick_i = 0
        self._log_every = max(1, int(round(hz)))  # ~1 log/seg en headless
        self._ref_gray: Optional[np.ndarray] = None
        self._ref_gray_proc: Optional[np.ndarray] = None   # CLAHE-processed reference
        self._ref_edges: Optional[np.ndarray] = None       # Canny edges of reference
        # Multi-template list: [center, left, right] variants loaded automatically
        # when <stem>_left.png / <stem>_right.png exist alongside the main template.
        # _match_template_multiscale / _match_template_edges iterate over ALL of them
        # and return the best match, so a crooked pickup is detected regardless of
        # which angle the robot happened to grab the pallet at.
        self._all_templates: list = []
        self._tpl_path = self._resolve_template_path(self._tpl_path)
        self._load_reference(self._tpl_path)

        # Per-pick-type profiles. Roller = the default params/template just
        # loaded; rack = its own template + thresholds (optional). Switched by
        # /robot_state in _robot_state_cb → _apply_logo_profile.
        self._roller_prof = {
            'path': self._tpl_path,
            'gray': self._ref_gray, 'gray_proc': self._ref_gray_proc,
            'edges': self._ref_edges, 'all_templates': self._all_templates,
            'roi': self._d_roi, 'match': self._d_match, 'stopscale': self._d_stopscale,
            'hold': self._d_hold,
        }
        self._rack_prof = None
        rack_tpl = str(self.get_parameter('rack_template_path').value)
        if rack_tpl:
            rack_tpl = self._resolve_template_path(rack_tpl)
            self.get_logger().info(f'Cargando perfil RACK desde: {rack_tpl}')
            rg = cv2.imread(rack_tpl, cv2.IMREAD_COLOR)
            if rg is not None:
                rgg = cv2.cvtColor(rg, cv2.COLOR_BGR2GRAY)
                rgg_proc = self._clahe.apply(rgg) if self._use_clahe else rgg
                rgg_edges = cv2.Canny(rgg, self._edge_low, self._edge_high)
                rack_all_tpl = self._load_template_variants(rack_tpl)
                self._rack_prof = {
                    'path': rack_tpl,
                    'gray': rgg, 'gray_proc': rgg_proc, 'edges': rgg_edges,
                    'all_templates': rack_all_tpl,
                    'roi': int(self.get_parameter('rack_roi_top_pct').value),
                    'match': int(round(float(self.get_parameter('rack_match_thr').value) * 100)),
                    'stopscale': int(round(float(self.get_parameter('rack_stop_scale').value) * 100)),
                    'hold': int(self.get_parameter('rack_hold_frames').value),
                }
                self.get_logger().info(
                    f'Perfil RACK: {rgg.shape[1]}x{rgg.shape[0]} de {rack_tpl} '
                    f'variantes={len(rack_all_tpl)} '
                    f'(roi={self._rack_prof["roi"]} stop_scale={self._rack_prof["stopscale"]/100:.2f} '
                    f'match_thr={self._rack_prof["match"]/100:.2f} hold={self._rack_prof["hold"]})')
            else:
                self.get_logger().warn(f'rack_template_path {rack_tpl!r} ilegible; sin perfil rack.')

        if self._show:
            try:
                self._build_windows()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'Sin GUI ({exc!r}); headless.')
                self._show = False

        self.create_timer(1.0 / max(1.0, hz), self._tick)
        self._start_keyboard_thread()
        n_tpl = len(self._all_templates)
        self.get_logger().info(
            f'logo_stop_debug listo  topic={self._image_topic}  '
            f'ref={"OK" if self._ref_gray is not None else "FALTA (presiona s)"}  '
            f'variantes={n_tpl} (s=centro a=izq d=der)  '
            f'escalas={self._scales[0]}..{self._scales[-1]}  gui={self._show}')

    # ------------------------------------------------------------------
    def _start_keyboard_thread(self) -> None:
        if not sys.stdin.isatty():
            return
        t = threading.Thread(target=self._keyboard_loop, daemon=True)
        t.start()

    def _keyboard_loop(self) -> None:
        """Lee teclas del terminal sin Enter (raw mode). s=centro a=izq d=der."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        self.get_logger().info('Teclado activo: s=centro  a=izq  d=der  q=salir')
        try:
            tty.setraw(fd)
            while rclpy.ok():
                ch = sys.stdin.read(1)
                if ch in ('q', '\x03', '\x1b'):  # q, Ctrl+C, ESC
                    rclpy.shutdown()
                    break
                if self._frame is None:
                    continue
                if ch == 's':
                    self._save_reference(self._frame, 'centro')
                elif ch == 'a':
                    self._save_reference(self._frame, 'izq')
                elif ch == 'd':
                    self._save_reference(self._frame, 'der')
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

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

    def _load_one_tpl(self, path: str) -> 'Optional[dict]':
        """Load one template image and return its preprocessed representations,
        or None if the file doesn't exist / can't be read."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_proc = self._clahe.apply(gray) if self._use_clahe else gray
        edges = cv2.Canny(gray, self._edge_low, self._edge_high)
        return {'gray': gray, 'gray_proc': gray_proc, 'edges': edges}

    def _load_template_variants(self, base_path: str) -> list:
        """Load the center template plus _left / _right angle variants.

        Convention:  <stem>_left<ext>  and  <stem>_right<ext>  alongside the
        main template.  Missing variants are silently skipped — center is always
        first.

        Captures workflow (GUI window):
            s = guardar centro  |  a = guardar izquierda  |  d = guardar derecha
        """
        import os as _os
        stem, ext = _os.path.splitext(base_path)
        paths = [
            ('centro', base_path),
            ('izq',    f'{stem}_left{ext}'),
            ('der',    f'{stem}_right{ext}'),
        ]
        templates = []
        for label, p in paths:
            t = self._load_one_tpl(p)
            if t is not None:
                t['label'] = label
                templates.append(t)
                if p != base_path:
                    self.get_logger().info(
                        f'  + variante [{label}]: {_os.path.basename(p)} '
                        f'({t["gray"].shape[1]}x{t["gray"].shape[0]})')
        return templates

    def _load_reference(self, path: str) -> bool:
        ref = cv2.imread(path, cv2.IMREAD_COLOR)
        if ref is None:
            self.get_logger().warn(f'No pude leer referencia en {path!r}.')
            return False
        gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        self._ref_gray = gray
        # CLAHE-processed version: used for template matching to be robust to
        # non-uniform illumination.
        self._ref_gray_proc = self._clahe.apply(gray) if self._use_clahe else gray
        # Canny edge map: illumination-invariant representation for mode 1 (edge).
        self._ref_edges = cv2.Canny(gray, self._edge_low, self._edge_high)
        # Load all angle variants (center + left + right).
        self._all_templates = self._load_template_variants(path)
        self.get_logger().info(
            f'Referencia {ref.shape[1]}x{ref.shape[0]} '
            f'clahe={"ON" if self._use_clahe else "OFF"} '
            f'variantes={len(self._all_templates)}')
        return True

    def _build_windows(self) -> None:
        cv2.namedWindow(WIN_T, cv2.WINDOW_NORMAL)
        # Controles globales + del metodo template en la ventana WIN_T.
        cv2.createTrackbar('ROI top %', WIN_T, self._d_roi, 90, _noop)
        cv2.createTrackbar('match thr x100', WIN_T, self._d_match, 100, _noop)
        cv2.createTrackbar('stop escala x100', WIN_T, self._d_stopscale, 130, _noop)
        cv2.createTrackbar('modo 0=tmpl 1=edge', WIN_T, self._d_mode, 1, _noop)
        cv2.createTrackbar('hold frames', WIN_T, self._d_hold, 30, _noop)

    def _tb(self, name: str, win: str, default: int) -> int:
        if not self._show:
            return default
        try:
            return int(cv2.getTrackbarPos(name, win))
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------
    def _apply_logo_profile(self, rack: bool) -> None:
        """Switch the active template + thresholds to the RACK profile (when
        /robot_state == rack_state) or the ROLLER/default profile. Only changes
        when needed; if no rack profile is configured, stays on roller."""
        if rack and self._rack_prof is None:
            return
        p = self._rack_prof if rack else self._roller_prof
        if self._ref_gray is p['gray']:
            return
        self._ref_gray = p['gray']
        self._ref_gray_proc = p.get('gray_proc', p['gray'])
        self._ref_edges = p.get('edges')
        self._all_templates = p.get('all_templates', [])
        self._tpl_path = p['path']
        self._d_roi = p['roi']; self._d_match = p['match']
        self._d_stopscale = p['stopscale']; self._d_hold = p['hold']
        self.get_logger().info(
            f'LOGO profile -> {"RACK" if rack else "ROLLER"} '
            f'variantes={len(self._all_templates)} '
            f'(roi={p["roi"]} stop_scale={p["stopscale"]/100:.2f} '
            f'match_thr={p["match"]/100:.2f} hold={p["hold"]})')

    def _robot_state_cb(self, msg: String) -> None:
        st = (msg.data or '').strip().upper()
        if st == self._robot_state:
            return
        self._robot_state = st
        self._apply_logo_profile(rack=(st == self._rack_state))

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
        """Multiscale template matching against ALL loaded angle variants.

        scene_gray must be CLAHE-preprocessed; each template in _all_templates
        already carries its gray_proc representation. Returns the best (score,
        scale, box) across every variant × scale combination so a pallet grabbed
        slightly left or right still registers a good match.
        """
        if not self._all_templates:
            return None
        Hs, Ws = scene_gray.shape
        best = None
        for tpl_dict in self._all_templates:
            tpl = tpl_dict['gray_proc']
            rh, rw = tpl.shape
            for s in self._scales:
                tw, th = int(rw * s), int(rh * s)
                if tw < 12 or th < 12 or tw > Ws or th > Hs:
                    continue
                t = cv2.resize(tpl, (tw, th))
                res = cv2.matchTemplate(scene_gray, t, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(res)
                if best is None or mv > best['score']:
                    best = {'score': float(mv), 'scale': float(s),
                            'box': (ml[0], ml[1], tw, th),
                            'variant': tpl_dict.get('label', '?')}
        return best

    def _match_template_edges(self, scene_gray):
        """Edge-based multiscale matching across ALL angle variants (mode 4).

        Canny edges are invariant to monotonic brightness changes, so this is
        the most lighting-robust path. Tries every variant × scale combination
        and returns the best match.
        """
        if not self._all_templates:
            return None
        scene_edges = cv2.Canny(scene_gray, self._edge_low, self._edge_high)
        scene_f = scene_edges.astype(np.float32)
        Hs, Ws = scene_edges.shape
        best = None
        for tpl_dict in self._all_templates:
            tpl_f = tpl_dict['edges'].astype(np.float32)
            rh, rw = tpl_f.shape
            for s in self._scales:
                tw, th = int(rw * s), int(rh * s)
                if tw < 12 or th < 12 or tw > Ws or th > Hs:
                    continue
                t = cv2.resize(tpl_f, (tw, th))
                res = cv2.matchTemplate(scene_f, t, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(res)
                if best is None or mv > best['score']:
                    best = {'score': float(mv), 'scale': float(s),
                            'box': (ml[0], ml[1], tw, th),
                            'variant': tpl_dict.get('label', '?')}
        return best

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        frame = self._frame
        if frame is None:
            return
        h, w = frame.shape[:2]

        # Gate por estado del SM: si active_states está configurado y el SM no
        # está en uno de ellos, no hay paro (publica False) y se salta el
        # matching (ahorra CPU).
        if self._active_states and self._robot_state not in self._active_states:
            self._hold = 0
            self._last_stop = False
            self._pub_stop.publish(Bool(data=False))
            return

        # Gate de armado: el SM arma justo antes del creep post-dock.
        # Hace matching igualmente (debug image actualizado) pero no para.
        if not self._armed:
            self._hold = 0
            self._pub_stop.publish(Bool(data=False))
            # (continúa para publicar debug image)

        roi_top_pct = self._tb('ROI top %', WIN_T, self._d_roi)
        match_thr = self._tb('match thr x100', WIN_T, self._d_match) / 100.0
        stop_scale = self._tb('stop escala x100', WIN_T, self._d_stopscale) / 100.0
        mode = self._tb('modo 0=tmpl 1=edge', WIN_T, self._d_mode)
        hold_n = self._tb('hold frames', WIN_T, self._d_hold)

        ry0 = int(h * max(0, min(90, roi_top_pct)) / 100.0)
        roi = frame[ry0:h, 0:w]
        scene_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Preprocessing: CLAHE normalizes local illumination before matching.
        # Applied to the scene ROI; the reference was already processed at load.
        scene_proc = self._clahe.apply(scene_gray) if self._use_clahe else scene_gray

        # ---- Template multiescala (con CLAHE) ----
        tm = self._match_template_multiscale(scene_proc)
        tmpl_ok = tm is not None and tm['score'] >= match_thr
        tmpl_stop = tmpl_ok and (tm['scale'] >= stop_scale)
        tmpl_cx = None
        if tm is not None:
            x, y, bw, bh = tm['box']
            tmpl_cx = x + bw / 2.0  # en coords de la ROI (x global == ROI x)

        # ---- Template sobre bordes Canny (modo 1: invariante a brillo) ----
        em = self._match_template_edges(scene_gray)
        edge_ok = em is not None and em['score'] >= match_thr
        edge_stop = edge_ok and (em['scale'] >= stop_scale)

        # ---- Decision + anti-rebote ----
        if mode == 0:
            raw = tmpl_stop
        else:  # mode 1: edge matching
            raw = edge_stop
        import time as _time
        in_grace = self._armed and (_time.monotonic() - self._arm_time) < self._arm_grace_s
        if self._armed and not in_grace:
            self._hold = self._hold + 1 if raw else 0
        elif in_grace:
            self._hold = 0
        should_stop = self._armed and not in_grace and self._hold >= max(1, hold_n)

        self._pub_stop.publish(Bool(data=bool(should_stop)))
        # Error de centrado (px) del metodo activo, para alineacion futura.
        if mode == 1:
            cx = (em['box'][0] + em['box'][2] / 2.0) if em is not None else None
        else:
            cx = tmpl_cx
        if cx is not None:
            self._pub_center.publish(Point(x=float(cx - w / 2.0), y=0.0, z=0.0))
        if should_stop != self._last_stop:
            sc = f'{tm["scale"]:.2f}' if tm else 'NA'
            scr = f'{tm["score"]:.2f}' if tm else 'NA'
            e_sc = f'{em["scale"]:.2f}' if em else 'NA'
            e_scr = f'{em["score"]:.2f}' if em else 'NA'
            self.get_logger().info(
                f'should_stop -> {should_stop}  '
                f'tmpl(score={scr} escala={sc} stop={tmpl_stop}) '
                f'edge(score={e_scr} escala={e_sc} stop={edge_stop})')
            self._last_stop = should_stop

        # Monitor periodico (util en headless mientras mueves el robot a mano).
        self._tick_i += 1
        if self._tick_i % self._log_every == 0:
            sc = f'{tm["scale"]:.2f}' if tm else 'NA'
            scr = f'{tm["score"]:.2f}' if tm else 'NA'
            e_sc = f'{em["scale"]:.2f}' if em else 'NA'
            e_scr = f'{em["score"]:.2f}' if em else 'NA'
            prof = 'RACK' if (self._rack_prof and self._ref_gray is self._rack_prof['gray']) else 'ROLLER'
            self.get_logger().info(
                f'[mon] prof={prof} armed={self._armed} grace={in_grace} '
                f'tmpl score={scr} escala={sc}/{stop_scale:.2f} '
                f'hold={self._hold}/{hold_n}  '
                f'roi_top={roi_top_pct}  '
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
            cv2.imshow(WIN_T, self._scaled(tview))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                rclpy.shutdown()
            elif key == ord('s'):
                self._save_reference(frame, 'centro')
            elif key == ord('a'):
                self._save_reference(frame, 'izq')
            elif key == ord('d'):
                self._save_reference(frame, 'der')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'GUI fallo ({exc!r}); headless.')
            self._show = False

    # ------------------------------------------------------------------
    def _save_reference(self, frame: np.ndarray, variant: str = 'centro') -> None:
        """Save the FULL frame for the given angle variant (no auto-crop).

        variant:
          'centro' → <stem>.<ext>          (tecla s)
          'izq'    → <stem>_left.<ext>     (tecla a — pallet chueco izquierda)
          'der'    → <stem>_right.<ext>    (tecla d — pallet chueco derecha)

        The full frame is saved so the user can crop it manually to get the
        exact logo region. After manually cropping and overwriting the file,
        press s/a/d again OR restart the node to reload.
        """
        import os as _os
        import re as _re
        # El launch usa get_package_share_directory → apunta a install/PKG/share/PKG/.
        # Para guardar en src/ (el original editable) remapeamos la ruta.
        tpl = self._tpl_path
        m = _re.match(r'(.+)/install/([^/]+)/share/\2/(.+)', tpl)
        if m:
            ws, pkg, rel = m.groups()
            src_cand = _os.path.join(ws, 'src', pkg, rel)
            if _os.path.isdir(_os.path.dirname(src_cand)):
                tpl = src_cand
        stem, ext = _os.path.splitext(tpl)
        if variant == 'izq':
            save_path = f'{stem}_left{ext}'
        elif variant == 'der':
            save_path = f'{stem}_right{ext}'
        else:
            save_path = tpl

        ok = cv2.imwrite(save_path, frame)
        self.get_logger().info(
            f'Frame [{variant}] {"guardado" if ok else "NO guardado"} en '
            f'{save_path} ({frame.shape[1]}x{frame.shape[0]}) — recorta manualmente el logo')
        if ok:
            self._load_reference(self._tpl_path)  # reload all variants

    def _enable_cb(self, msg: Bool) -> None:
        import time as _time
        was = self._armed
        self._armed = bool(msg.data)
        if self._armed and not was:
            self._hold = 0
            self._arm_time = _time.monotonic()
            is_rack = self._robot_state == self._rack_state
            self._arm_grace_s = 3.0 if is_rack else 0.0
            self.get_logger().info(
                f'Logo stop ARMADO ({"RACK" if is_rack else "ROLLER"} state={self._robot_state!r}) '
                f'— gracia {self._arm_grace_s:.1f}s antes de contar')
        elif not self._armed:
            self.get_logger().info('Logo stop desarmado')

    def _save_ref_cb(self, msg: String) -> None:
        """Topic callback para guardar foto sin ventana GUI.

        Publica: ros2 topic pub --once /logo_stop/save_ref std_msgs/msg/String 'data: centro'
        Valores: centro | izq | der
        """
        variant = msg.data.strip().lower()
        if variant not in ('centro', 'izq', 'der'):
            self.get_logger().warn(
                f'/logo_stop/save_ref: variante desconocida "{variant}" — usa centro|izq|der')
            return
        if self._frame is None:
            self.get_logger().warn('/logo_stop/save_ref: sin frame aún, espera la cámara.')
            return
        self._save_reference(self._frame, variant)

    def _scaled(self, img):
        if self._scale and self._scale != 1.0:
            return cv2.resize(img, None, fx=self._scale, fy=self._scale,
                              interpolation=cv2.INTER_NEAREST)
        return img

    def _draw_template(self, frame, ry0, tm, ok, stop, thr, stop_scale,
                       should_stop, hold_n):
        out = frame.copy()
        cv2.line(out, (0, ry0), (out.shape[1], ry0), CYAN, 1)
        n_tpl = len(self._all_templates)
        if tm is not None:
            x, y, bw, bh = tm['box']
            color = GREEN if stop else (YELLOW if ok else RED)
            cv2.rectangle(out, (x, y + ry0), (x + bw, y + bh + ry0), color, 2)
            cv2.line(out, (x + bw // 2, y + ry0),
                     (x + bw // 2, y + bh + ry0), color, 1)
            vname = tm.get('variant', '?')
            _text(out, f'score {tm["score"]:.2f}/{thr:.2f}  [{vname}]', (6, 16))
            _text(out, f'escala {tm["scale"]:.2f}  paro>={stop_scale:.2f}', (6, 32))
            cx = x + bw / 2.0
            _text(out, f'centro err {cx - out.shape[1]/2:+.0f}px', (6, 48))
            _text(out, f'variantes={n_tpl}  s=centro a=izq d=der', (6, 64), CYAN)
            _text(out, f'modo 0=tmpl 1=edge', (6, 80), CYAN)
        else:
            _text(out, f'SIN REF ({n_tpl} var)  s=centro a=izq d=der', (6, 20), RED)
        _tag(out, 'TMPL', stop, ok, (out.shape[1] - 96, 14))
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
