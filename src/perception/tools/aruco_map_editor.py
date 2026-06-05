#!/usr/bin/env python3
"""Servidor local para el editor visual del mapa de marcadores ArUco.

Sirve `aruco_map_editor.html` y expone dos endpoints que leen/escriben
directamente `src/perception/config/aruco_map.yaml`:

    GET  /api/load  -> {track, markers, obstacles}  (JSON)
    POST /api/save  <- {track, markers, obstacles}  -> reescribe el YAML

Sin dependencias extra: usa la stdlib + PyYAML (ya presente en el proyecto).

Uso:
    python3 src/perception/tools/aruco_map_editor.py
    # abre el navegador en http://127.0.0.1:8770 automaticamente

El YAML resultante es el que consume `aruco_localization` (clave `markers`);
`track` y `obstacles` son metadatos que el nodo de localizacion ignora.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, 'aruco_map_editor.html')
MAP_YAML = os.path.abspath(os.path.join(HERE, '..', 'config', 'aruco_map.yaml'))
PORT = int(os.environ.get('ARUCO_EDITOR_PORT', '8770'))


def _read_map() -> dict:
    """Lee el YAML y normaliza a {track, markers, obstacles}."""
    if not os.path.exists(MAP_YAML):
        return {'track': {'x': 3.67, 'y': 4.85}, 'markers': [], 'obstacles': []}
    with open(MAP_YAML, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    raw = data.get('markers', []) if isinstance(data, dict) else (data or [])
    markers = []
    for m in raw:
        yaw = m.get('yaw_deg', m.get('theta_deg'))
        if yaw is None and 'theta' in m:
            yaw = math.degrees(float(m['theta']))
        markers.append({
            'id': int(m['id']),
            'x': float(m['x']),
            'y': float(m['y']),
            'z': float(m.get('z', 0.10)),
            'yaw_deg': float(yaw or 0.0),
            'mount': str(m.get('mount', 'wall')),
        })
    track = data.get('track', {'x': 3.67, 'y': 4.85}) if isinstance(data, dict) else {'x': 3.67, 'y': 4.85}
    obstacles = data.get('obstacles', []) if isinstance(data, dict) else []
    return {'track': track, 'markers': markers, 'obstacles': obstacles}


def _write_map(payload: dict) -> None:
    """Reescribe aruco_map.yaml en el formato legible que usa el proyecto."""
    track = payload.get('track', {'x': 3.67, 'y': 4.85})
    markers = payload.get('markers', [])
    obstacles = payload.get('obstacles', [])

    lines = [
        '# =============================================================================',
        '# Mapa de marcadores ArUco de localizacion  (editado con aruco_map_editor)',
        '# =============================================================================',
        '# Frame `map` (REP-103): X adelante, Y izquierda, Z arriba.',
        '# Markers VERTICALES (en muros / costados de obstaculos). x,y = centro [m];',
        '# z = altura del centro sobre el piso [m]; yaw_deg = direccion a la que MIRA',
        '# la cara (su normal) [grados]; mount = wall (vertical) | floor (plano).',
        '# Diccionario DICT_ARUCO_ORIGINAL (5x5), lado 9 cm.',
        '# =============================================================================',
        '',
        f'track: {{x: {float(track["x"]):.3f}, y: {float(track["y"]):.3f}}}',
        '',
        'markers:',
    ]
    for m in markers:
        lines.append(
            f'  - {{id: {int(m["id"])}, x: {float(m["x"]):.3f}, '
            f'y: {float(m["y"]):.3f}, z: {float(m.get("z", 0.10)):.3f}, '
            f'yaw_deg: {float(m.get("yaw_deg", 0.0)):.1f}, '
            f'mount: {str(m.get("mount", "wall"))}}}'
        )
    if obstacles:
        lines += ['', '# Solo referencia visual; aruco_localization los ignora.', 'obstacles:']
        for o in obstacles:
            lines.append(
                f'  - {{x: {float(o["x"]):.3f}, y: {float(o["y"]):.3f}, '
                f'w: {float(o["w"]):.3f}, h: {float(o["h"]):.3f}}}'
            )
    text = '\n'.join(lines) + '\n'

    os.makedirs(os.path.dirname(MAP_YAML), exist_ok=True)
    if os.path.exists(MAP_YAML):
        shutil.copyfile(MAP_YAML, MAP_YAML + '.bak')  # respaldo del anterior
    with open(MAP_YAML, 'w', encoding='utf-8') as f:
        f.write(text)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(HTML, 'rb') as f:
                self._send(200, f.read(), 'text/html; charset=utf-8')
        elif self.path == '/api/load':
            self._send(200, json.dumps(_read_map()))
        else:
            self._send(404, json.dumps({'error': 'not found'}))

    def do_POST(self):
        if self.path != '/api/save':
            self._send(404, json.dumps({'error': 'not found'}))
            return
        try:
            n = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(n) or b'{}')
            _write_map(payload)
            self._send(200, json.dumps({'ok': True, 'path': MAP_YAML}))
        except Exception as e:  # noqa: BLE001
            self._send(200, json.dumps({'ok': False, 'error': str(e)}))

    def log_message(self, *args):  # silencia el log por request
        pass


def main():
    if not os.path.exists(HTML):
        raise SystemExit(f'No encuentro el HTML: {HTML}')
    url = f'http://127.0.0.1:{PORT}'
    print(f'Editor de mapa ArUco -> {url}')
    print(f'Escribe en: {MAP_YAML}')
    try:
        webbrowser.open(url)
    except Exception:
        pass
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
