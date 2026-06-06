#!/usr/bin/env python3
"""extract_static_layout.py — Saca la geometría estática (paredes, racks, rollers,
camiones) del gemelo digital `almacen_racks.world` y emite el layout que el SLAM
estampa en el grid al anclar.

Emite DOS YAMLs (mismo formato, distinto frame):
  perception/config/static_layout_sim.yaml   — frame del .world (= canónico SIM 4.85×3.65). EXACTO.
  perception/config/static_layout_real.yaml  — frame canónico REAL (aruco_map.yaml, 3.65×4.85).
       Transformado por best-fit (Umeyama robusto) sobre los marcadores compartidos.
       Los racks se construyeron a la medida del .world; el desfase (~0.3-0.5 m) viene
       del survey de marcadores → al transformar por los MISMOS marcadores, el layout
       queda CONSISTENTE con la creencia del robot (que ancla a aruco_map.yaml).

Formato de salida (cada obstáculo, footprint 2D en el frame correspondiente):
  - {name, x, y, sx, sy, yaw_deg}   # centro (m), tamaño (m), giro (grados)

Uso:  python3 scripts/extract_static_layout.py [--plot [out.png]]
"""
import argparse
import math
import os
import re
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD = os.path.join(ROOT, 'src/description/worlds/almacen_racks.world')
ARUCO_REAL = os.path.join(ROOT, 'src/perception/config/aruco_map.yaml')
OUT_SIM = os.path.join(ROOT, 'src/perception/config/static_layout_sim.yaml')
OUT_REAL = os.path.join(ROOT, 'src/perception/config/static_layout_real.yaml')

# Modelos a ESTAMPAR (obstáculos que el LiDAR ve y el robot esquiva). Se excluyen
# markers (aruco_*), ground_plane y dinteles (van por ENCIMA del LiDAR).
INCLUDE_PREFIXES = ('pared_', 'dock_', 'rack_', 'roller_')
EXCLUDE_PREFIXES = ('aruco_', 'ground_plane', 'dintel_')
LIDAR_Z_MAX = 0.25   # m; un obstáculo cuenta si su BASE está por debajo de esto (toca el piso)

# Correcciones manuales post-Umeyama para el frame REAL.
# Origen: medición física en pista. Formato: {prefijo_nombre: {dx, dy}} en metros.
# rack_C1/C2: cara norte del bay superior debe estar a 130 cm de la pared norte
# (la transformación gz→real los dejaba en ~123 cm → corrección -0.07 m en Y).
MANUAL_REAL_CORRECTIONS = {
    'rack_C1_': {'dx': 0.0, 'dy': -0.07},
    'rack_C2_': {'dx': 0.0, 'dy': -0.07},
}


def parse_world(path):
    """Devuelve [{name, x, y, yaw, sx, sy, z, sz}] de los modelos con pose + box."""
    txt = open(path).read()
    out = []
    # Cada bloque <model name="..."> ... </model>. El primer <pose> es la pose del
    # modelo; el primer <box><size> es el footprint.
    for mm in re.finditer(r'<model name="([^"]+)">(.*?)</model>', txt, re.S):
        name, body = mm.group(1), mm.group(2)
        pm = re.search(r'<pose>([^<]+)</pose>', body)
        sm = re.search(r'<box><size>([^<]+)</size></box>', body)
        if not pm or not sm:
            continue
        p = [float(v) for v in pm.group(1).split()]
        s = [float(v) for v in sm.group(1).split()]
        x, y, z = p[0], p[1], p[2]
        yaw = p[5] if len(p) >= 6 else 0.0
        sx, sy, sz = s[0], s[1], s[2]
        out.append(dict(name=name, x=x, y=y, yaw=yaw, sx=sx, sy=sy, z=z, sz=sz))
    return out


def is_obstacle(m):
    n = m['name']
    if any(n.startswith(p) for p in EXCLUDE_PREFIXES):
        return False
    if not any(n.startswith(p) for p in INCLUDE_PREFIXES):
        return False
    # Filtro de altura: la base del box debe tocar (o casi) el piso → el LiDAR lo ve.
    return (m['z'] - m['sz'] / 2.0) < LIDAR_Z_MAX


def aruco_world_positions(path):
    """Posiciones (gz/sim frame) de los modelos aruco_<id> del .world."""
    txt = open(path).read()
    d = {}
    for mm in re.finditer(r'<model name="aruco_(\d+)">.*?<pose>([^<]+)</pose>', txt, re.S):
        p = mm.group(2).split()
        d[int(mm.group(1))] = (float(p[0]), float(p[1]))
    return d


def fit2d_robust(src, dst, drop_thresh=0.6, min_keep=6):
    """Umeyama 2D (rot+trans) con drop-worst de outliers. Devuelve (R, t, kept_idx, rmse)."""
    src = [np.asarray(p, float) for p in src]
    dst = [np.asarray(p, float) for p in dst]
    idx = list(range(len(src)))
    while True:
        S = np.array([src[i] for i in idx]); D = np.array([dst[i] for i in idx])
        sc = S.mean(0); dc = D.mean(0)
        H = (D - dc).T @ (S - sc)
        yaw = math.atan2(H[1, 0] - H[0, 1], H[0, 0] + H[1, 1])
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]]); t = dc - R @ sc
        res = np.array([np.linalg.norm(R @ src[i] + t - dst[i]) for i in idx])
        rmse = float(np.sqrt((res ** 2).mean()))
        k = int(np.argmax(res))
        if res[k] <= drop_thresh or len(idx) <= min_keep:
            return R, t, idx, rmse
        del idx[k]


def main_impl(args):
    models = parse_world(WORLD)
    obstacles = [m for m in models if is_obstacle(m)]
    print(f'modelos: {len(models)} | obstáculos a estampar: {len(obstacles)}')

    # --- SIM: frame del .world directo ---
    sim_layout = [dict(name=m['name'], x=round(m['x'], 4), y=round(m['y'], 4),
                       sx=round(m['sx'], 4), sy=round(m['sy'], 4),
                       yaw_deg=round(math.degrees(m['yaw']), 2)) for m in obstacles]
    yaml.safe_dump({'track': {'x': 4.85, 'y': 3.65}, 'obstacles': sim_layout},
                   open(OUT_SIM, 'w'), sort_keys=False, default_flow_style=False)
    print(f'escrito {OUT_SIM} ({len(sim_layout)} obstáculos)')

    # --- REAL: transforma gz -> real por best-fit sobre marcadores ---
    gz = aruco_world_positions(WORLD)
    real = {m['id']: (m['x'], m['y']) for m in yaml.safe_load(open(ARUCO_REAL))['markers']}
    common = sorted(set(gz) & set(real))
    R, t, kept, rmse = fit2d_robust([gz[i] for i in common], [real[i] for i in common])
    tyaw = math.atan2(R[1, 0], R[0, 0])
    print(f'transform gz->real: {len(kept)}/{len(common)} markers, yaw={math.degrees(tyaw):.1f}°, '
          f't=({t[0]:.3f},{t[1]:.3f}), rmse={rmse:.3f} m (outliers descartados: '
          f'{sorted(set(common)-set(common[i] for i in kept))})')
    real_layout = []
    for m in obstacles:
        c = R @ np.array([m['x'], m['y']]) + t
        real_layout.append(dict(name=m['name'], x=round(float(c[0]), 4), y=round(float(c[1]), 4),
                                sx=round(m['sx'], 4), sy=round(m['sy'], 4),
                                yaw_deg=round(math.degrees(m['yaw'] + tyaw), 2)))
    # Aplica correcciones manuales de survey al frame real
    for obs in real_layout:
        for prefix, delta in MANUAL_REAL_CORRECTIONS.items():
            if obs['name'].startswith(prefix):
                obs['x'] = round(obs['x'] + delta.get('dx', 0.0), 4)
                obs['y'] = round(obs['y'] + delta.get('dy', 0.0), 4)

    yaml.safe_dump({'track': {'x': 3.65, 'y': 4.85},
                    'note': f'gz->real best-fit rmse={rmse:.3f} m; correcciones manuales aplicadas',
                    'obstacles': real_layout},
                   open(OUT_REAL, 'w'), sort_keys=False, default_flow_style=False)
    print(f'escrito {OUT_REAL} ({len(real_layout)} obstáculos)')
    return real_layout, real


def plot_map(real_layout, arucos, out_png):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    TRACK_W, TRACK_H = 3.65, 4.85

    # Clasificación de obstáculos por tipo para colorear
    COLOR_MAP = {
        'pared_': '#555555',
        'dock_':  '#888888',
        'rack_':  '#c8a87a',
        'roller_': '#aaccee',
    }

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(-0.15, TRACK_W + 0.15)
    ax.set_ylim(-0.8, TRACK_H + 0.15)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f0eb')
    ax.set_xlabel('X — Oeste → Este [m]', fontsize=10)
    ax.set_ylabel('Y — Sur → Norte [m]', fontsize=10)
    ax.set_title('Mapa 2D — ArUcos (rojo=pared · naranja=rack · gris=pendiente)', fontsize=11)

    # Dibujar pista
    from matplotlib.patches import Rectangle
    track_rect = Rectangle((0, 0), TRACK_W, TRACK_H, linewidth=2.5,
                            edgecolor='black', facecolor='none')
    ax.add_patch(track_rect)

    # Dibujar obstáculos
    for obs in real_layout:
        cx, cy = obs['x'], obs['y']
        sx, sy = obs['sx'], obs['sy']
        yaw = math.radians(obs['yaw_deg'])
        color = '#aaaaaa'
        for prefix, c in COLOR_MAP.items():
            if obs['name'].startswith(prefix):
                color = c
                break

        # Rectángulo rotado
        corners = np.array([[-sx/2, -sy/2], [sx/2, -sy/2],
                             [sx/2,  sy/2], [-sx/2,  sy/2]])
        R = np.array([[math.cos(yaw), -math.sin(yaw)],
                      [math.sin(yaw),  math.cos(yaw)]])
        corners = (R @ corners.T).T + np.array([cx, cy])
        poly = plt.Polygon(corners, closed=True, facecolor=color,
                           edgecolor='#444444', linewidth=0.5, alpha=0.85)
        ax.add_patch(poly)

    # ArUcos en pared edificio vs rack
    RACK_ARUCO_IDS = {13, 19, 20, 25, 28, 31, 41}
    for m in arucos.values() if isinstance(arucos, dict) else arucos:
        mid = m['id']
        mx, my = m['x'], m['y']
        color = '#e07020' if mid in RACK_ARUCO_IDS else '#cc2222'
        ax.plot(mx, my, 's', color=color, markersize=7, zorder=5)

        # Distancias desde paredes
        dn = round((TRACK_H - my) * 100)
        ds = round(my * 100)
        de = round((TRACK_W - mx) * 100)
        do_ = round(mx * 100)
        label = (f'#{mid}\n'
                 f'↑N {dn}cm ↓S {ds}cm\n'
                 f'→E {de}cm ←O {do_}cm')
        offx, offy = 0.07, 0.07
        if mx > TRACK_W * 0.75:
            offx = -0.07
        ax.annotate(label, (mx, my), xytext=(mx + offx, my + offy),
                    fontsize=5.5, color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, alpha=0.8))

    # Leyenda
    legend_elements = [
        mpatches.Patch(facecolor='#c8a87a', edgecolor='#444', label='Racks'),
        mpatches.Patch(facecolor='#aaccee', edgecolor='#444', label='Rollers'),
        mpatches.Patch(facecolor='#555555', edgecolor='#444', label='Paredes/Docks'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#cc2222',
                   markersize=8, label='ArUco pared edificio'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#e07020',
                   markersize=8, label='ArUco pared rack'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    ax.grid(True, linewidth=0.4, color='#cccccc')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'mapa guardado en {out_png}')
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', nargs='?', const='map_real.png', metavar='OUT.png',
                    help='Genera imagen del mapa real con ArUcos')
    args = ap.parse_args()
    real_layout, arucos_real = main_impl(args)
    if args.plot:
        aruco_data = yaml.safe_load(open(ARUCO_REAL))['markers']
        plot_map(real_layout, aruco_data, args.plot)


if __name__ == '__main__':
    main()
