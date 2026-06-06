#!/usr/bin/env python3
"""Regenera racks y rollers en almacen_racks.world desde el layout REAL medido.

Convención de entrada = la del usuario (igual que aruco_map.yaml):
    x = Este [0..3.65],  y = Norte [0..4.85],  origen en la esquina SW.
    Las zonas se dan como RECTÁNGULOS esquina (x,y) + tamaño (w,h) en metros
    (tal como los exporta el editor de obstáculos).

Conversión a Gazebo (gz):  gz_x = y_user,  gz_y = MAP_W - x_user.

- Cada zona de rack se llena con una rejilla nx×ny de racks chicos.
- Los rollers son cajas (17×35 cm) pegadas a la pared Norte.

Idempotente: borra los modelos rack_*/roller_* existentes y reinserta.
"""
import os
import re
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..')
WORLD = os.path.join(BASE, 'src', 'description', 'worlds', 'almacen_racks.world')
RACK_CFG = os.path.join(BASE, 'src', 'description', 'config', 'rack_groups.yaml')
ROLL_CFG = os.path.join(BASE, 'src', 'description', 'config', 'roller_groups.yaml')
MAP_W = 3.65

_rack = yaml.safe_load(open(RACK_CFG))
_roll = yaml.safe_load(open(ROLL_CFG))
RACK_H = float(_rack.get('rack_height', 0.20))
ROLL_H = float(_roll.get('roller_height', 0.20))
# (nombre, esquina_x, esquina_y, w, h, nx, ny) en coords usuario
RACK_ZONES = [(z['name'], z['x'], z['y'], z['w'], z['h'], z['nx'], z['ny'])
              for z in _rack['zones']]
ROLL_DEPTH = float(_roll.get('depth', 0.35))
ROLL_NORTH = float(_roll.get('north_y', 4.85))
ROLLERS = [(r['name'], r['x_ini'], r['x_fin']) for r in _roll['rollers']]


def u2gz(ux, uy):
    return uy, MAP_W - ux


def box_model(name, ucx, ucy, usx, usy, h, z_h, rgba):
    """Caja axis-aligned. (ucx,ucy)=centro usuario; (usx,usy)=tamaño usuario X,Y."""
    gx, gy = u2gz(ucx, ucy)
    gsx, gsy = usy, usx          # X_user→gz_y, Y_user→gz_x  ⇒ size se intercambia
    return (
        f'    <model name="{name}"><static>true</static>\n'
        f'      <pose>{gx:.4f} {gy:.4f} {z_h/2:.4f} 0 0 0</pose>\n'
        f'      <link name="link">\n'
        f'        <visual name="vis">\n'
        f'          <geometry><box><size>{gsx:.4f} {gsy:.4f} {h:.4f}</size></box></geometry>\n'
        f'          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>\n'
        f'        </visual>\n'
        f'        <collision name="col">\n'
        f'          <geometry><box><size>{gsx:.4f} {gsy:.4f} {h:.4f}</size></box></geometry>\n'
        f'        </collision>\n'
        f'      </link></model>'
    )


blocks = []
# Racks: rejilla nx×ny llenando cada zona
for zone, x0, y0, w, h, nx, ny in RACK_ZONES:
    cw, ch = w / nx, h / ny
    idx = 0
    for i in range(nx):
        for j in range(ny):
            idx += 1
            cx = x0 + (i + 0.5) * cw
            cy = y0 + (j + 0.5) * ch
            blocks.append(box_model(f'rack_{zone}_{idx}', cx, cy, cw, ch, RACK_H,
                                    RACK_H, '0.20 0.30 0.70 1'))

# Rollers: cajas pegadas a la pared Norte
for name, xi, xf in ROLLERS:
    cx = (xi + xf) / 2.0
    sx = xf - xi
    cy = ROLL_NORTH - ROLL_DEPTH / 2.0
    blocks.append(box_model(f'roller_{name}', cx, cy, sx, ROLL_DEPTH, ROLL_H,
                            ROLL_H, '0.70 0.45 0.15 1'))

content = open(WORLD).read()
# Borrar modelos rack_*/roller_* existentes
content = re.sub(r'\s*<model name="(rack|roller)_[^"]*".*?</model>', '', content,
                 flags=re.DOTALL)
# Insertar antes de </world>
content = content.replace('  </world>', '\n'.join(blocks) + '\n  </world>')
open(WORLD, 'w').write(content)

print(f"OK: {len(blocks)} modelos (racks+rollers) regenerados en {os.path.basename(WORLD)}")
for zone, x0, y0, w, h, nx, ny in RACK_ZONES:
    print(f"  rack {zone}: {nx}x{ny}={nx*ny} racks, zona ({x0},{y0}) {w}x{h} m")
print(f"  rollers: {len(ROLLERS)} (17?x35 cm, pared Norte y={ROLL_NORTH})")
