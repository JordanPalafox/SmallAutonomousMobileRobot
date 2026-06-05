#!/usr/bin/env python3
"""Vista previa 2D de marcadores ArUco sobre el mapa del almacén."""

import os
import sys
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ── Dimensiones del mapa (sistema ArUco: x=Este, y=Norte) ──
MAP_W = 3.65   # E-W [m]
MAP_H = 4.85   # N-S [m]

# ── Transformación rack/roller (cx=N-S, cy=E-W desde Este) → aruco (x=E, y=N) ──
def r2a(cx, cy):
    return MAP_W - cy, cx


# ── Marcadores ArUco ──
MARKERS = [
    {'id':  5, 'x': 3.650, 'y': 3.980, 'z': 0.400, 'yaw': 180.0},
    {'id':  7, 'x': 0.000, 'y': 4.515, 'z': 0.600, 'yaw':   0.0},
    {'id': 13, 'x': 2.400, 'y': 4.850, 'z': 0.600, 'yaw': 270.0},
    {'id': 17, 'x': 0.505, 'y': 4.850, 'z': 0.800, 'yaw': 270.0},
    {'id': 19, 'x': 1.170, 'y': 3.780, 'z': 0.450, 'yaw':  90.0},   # C1 cara Norte
    {'id': 20, 'x': 2.400, 'y': 3.780, 'z': 0.100, 'yaw':  90.0},   # C2 cara Norte
    {'id': 21, 'x': 3.650, 'y': 0.790, 'z': 0.600, 'yaw': 180.0},
    {'id': 22, 'x': 3.530, 'y': 0.000, 'z': 0.500, 'yaw':  90.0},
    {'id': 25, 'x': 2.400, 'y': 2.730, 'z': 0.500, 'yaw': 270.0},   # C2 cara Sur
    {'id': 28, 'x': 2.420, 'y': 1.500, 'z': 0.500, 'yaw':   0.0},   # interior
    {'id': 30, 'x': 1.620, 'y': 0.000, 'z': 0.450, 'yaw':  90.0},
    {'id': 31, 'x': 1.360, 'y': 1.520, 'z': 0.650, 'yaw': 180.0},   # interior
    {'id': 37, 'x': 0.000, 'y': 0.950, 'z': 0.550, 'yaw':   0.0},
    {'id': 38, 'x': 0.400, 'y': 0.000, 'z': 0.550, 'yaw':  90.0},
    {'id': 41, 'x': 1.170, 'y': 2.730, 'z': 0.400, 'yaw': 270.0},   # C1 cara Sur
]

# Categorías de marcadores
WALL_IDS      = {5, 7, 13, 17, 21, 22, 30, 37, 38}       # paredes del edificio
RACK_WALL_IDS = {19, 41, 20, 25, 31, 28}                  # paredes de racks C1/C2/B

BASE = os.path.join(os.path.dirname(__file__), '..')

# ── Figura ──
fig, ax = plt.subplots(figsize=(8, 11))
ax.set_aspect('equal')
ax.set_xlim(-0.35, MAP_W + 0.35)
ax.set_ylim(-0.35, MAP_H + 0.35)
ax.set_xlabel('X — Oeste → Este [m]', fontsize=9)
ax.set_ylabel('Y — Sur → Norte [m]', fontsize=9)
ax.set_title('Mapa 2D — ArUcos (rojo=pared · naranja=rack · gris=pendiente)', fontsize=9)

# ── Suelo ──
ax.add_patch(patches.Rectangle((0, 0), MAP_W, MAP_H,
                                linewidth=2, edgecolor='black', facecolor='#f7f3ec'))

# ── Racks ──
rack_path = os.path.join(BASE, 'src', 'slam', 'config', 'rack_groups.yaml')
with open(rack_path) as f:
    rack_data = yaml.safe_load(f)

rack_L = rack_data['rack_dims']['largo']
rack_P = rack_data['rack_dims']['prof']

for grp, items in rack_data['groups'].items():
    gxs, gys = [], []
    for r in items:
        ax_x, ax_y = r2a(r['cx'], r['cy'])
        gxs.append(ax_x); gys.append(ax_y)
        yaw_n = r['yaw'] % 180
        rw = rack_P if yaw_n == 0 else rack_L   # E-W extent
        rh = rack_L if yaw_n == 0 else rack_P   # N-S extent
        ax.add_patch(patches.Rectangle(
            (ax_x - rw / 2, ax_y - rh / 2), rw, rh,
            linewidth=0.7, edgecolor='saddlebrown', facecolor='#cda97e', alpha=0.75))
    ax.text(np.mean(gxs), np.mean(gys), grp,
            ha='center', va='center', fontsize=7,
            color='saddlebrown', fontweight='bold')

# ── Rollers ──
roll_path = os.path.join(BASE, 'src', 'slam', 'config', 'roller_groups.yaml')
with open(roll_path) as f:
    roll_data = yaml.safe_load(f)

roll_L = roll_data['roller_dims']['largo']
roll_A = roll_data['roller_dims']['ancho']

for grp, items in roll_data['groups'].items():
    gxs, gys = [], []
    for r in items:
        ax_x, ax_y = r2a(r['cx'], r['cy'])
        gxs.append(ax_x); gys.append(ax_y)
        yaw_n = r['yaw'] % 180
        rw = roll_A if yaw_n == 0 else roll_L   # E-W
        rh = roll_L if yaw_n == 0 else roll_A   # N-S
        ax.add_patch(patches.Rectangle(
            (ax_x - rw / 2, ax_y - rh / 2), rw, rh,
            linewidth=0.5, edgecolor='steelblue', facecolor='#9fc8e8', alpha=0.85))
    ax.text(np.mean(gxs), np.mean(gys), grp,
            ha='center', va='center', fontsize=6, color='navy', fontweight='bold')

# ── Marcadores ArUco ──
# Convención yaw: 0=+X(Este), 90=+Y(Norte), 180=-X(Oeste), 270=-Y(Sur)
ARROW = 0.12

for m in MARKERS:
    mid = m['id']
    mx, my = m['x'], m['y']

    if mid in WALL_IDS:
        color, arrow_color, fcolor = 'red', 'red', 'darkred'
        ec = 'darkred'; alpha = 1.0
        label = f'#{mid}\nz={m["z"]}m'
        italic = False
    elif mid in RACK_WALL_IDS:
        color, arrow_color, fcolor = 'darkorange', 'darkorange', '#7a3a00'
        ec = 'darkorange'; alpha = 1.0
        label = f'#{mid}\nz={m["z"]}m'
        italic = False
    else:
        color, arrow_color, fcolor = '#888888', '#888888', '#555555'
        ec = 'gray'; alpha = 0.45
        label = f'#{mid}'
        italic = True

    yaw_rad = math.radians(m['yaw'])
    dx = math.cos(yaw_rad) * ARROW
    dy = math.sin(yaw_rad) * ARROW

    ax.plot(mx, my, 's', color=color, markersize=9, zorder=6, alpha=alpha)

    if mid in WALL_IDS or mid in RACK_WALL_IDS:
        ax.annotate('', xy=(mx + dx, my + dy), xytext=(mx, my),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.6))

    bbox = dict(boxstyle='round,pad=0.15', fc='white', ec=ec,
                alpha=0.85 if not italic else 0.6)
    ax.text(mx + 0.07, my + 0.05, label,
            fontsize=6.5, color=fcolor, zorder=7,
            bbox=bbox, style='italic' if italic else 'normal')

# ── Etiquetas de paredes ──
for label, x, y, ha, va in [
    ('Pared Norte', MAP_W/2,  MAP_H+0.18, 'center', 'bottom'),
    ('Pared Sur',   MAP_W/2,      -0.18,  'center', 'top'),
    ('Pared Oeste',     -0.18, MAP_H/2,   'right',  'center'),
    ('Pared Este',  MAP_W+0.18, MAP_H/2,  'left',   'center'),
]:
    ax.text(x, y, label, ha=ha, va=va, fontsize=8, color='#333333',
            rotation=0 if 'Norte' in label or 'Sur' in label else 90)

# ── Leyenda ──
handles = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red',        markersize=9, label='ArUco pared edificio'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange',  markersize=9, label='ArUco pared rack (C1/C2/B)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#888888',     markersize=9, alpha=0.5, label='ArUco pendiente'),
    patches.Patch(facecolor='#cda97e', edgecolor='saddlebrown', label='Racks'),
    patches.Patch(facecolor='#9fc8e8', edgecolor='steelblue',   label='Rollers'),
]
ax.legend(handles=handles, loc='lower right', fontsize=7.5, framealpha=0.9)

# ── Grid ──
ax.grid(True, linestyle='--', alpha=0.25)
ax.set_xticks(np.arange(0, MAP_W + 0.01, 0.5))
ax.set_yticks(np.arange(0, MAP_H + 0.01, 0.5))

plt.tight_layout()
out = '/tmp/aruco_map_preview.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Guardado en {out}")
plt.show()
