#!/usr/bin/env python3
"""
Inserta modelos ArUco en almacen_racks.world.
Idempotente: elimina aruco_ existentes antes de agregar los nuevos.

Sistemas de coordenadas:
  ArUco YAML : x=Este, y=Norte, origen=SW, yaw=CCW desde Este
  Gazebo world: X=Norte, Y=Este→Oeste (Y=0 en Pared Este)
  Conversión: gz_x=aruco_y, gz_y=MAP_W-aruco_x
  Rotación SDF: roll=π, pitch=-π/2, yaw=aruco_yaw_rad-π/2
    → cara del marcador apunta en la dirección correcta y la imagen queda vertical.
"""

import os
import re
import math
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.join(SCRIPT_DIR, '..')
WORLD_PATH = os.path.join(BASE, 'src', 'description', 'worlds', 'almacen_racks.world')
YAML_PATH  = os.path.join(BASE, 'src', 'slam', 'config', 'aruco_markers.yaml')
IMG_DIR    = os.path.join(BASE, 'src', 'description', 'textures', 'arucos')

MAP_W  = 3.65   # ancho E-W del mapa en coords ArUco [m]
SIDE   = 0.09   # lado del marcador [m]
# Empuje del marcador hacia ADENTRO (a lo largo de su normal) SOLO para los
# muros del perímetro: sus coords (borde interno 3.65x4.85) caen en la cara
# EXTERIOR del muro del modelo (grosor 0.03 m). Con INSET=0.03 el marcador queda
# FLUSH en la cara interior (su trasera embebida en el muro -> desde afuera no se
# ve el espejo de la cara trasera). Los racks NO se empujan (ya están en su cara).
INSET  = 0.0    # paredes reescaladas a 3.65x4.85 -> marcadores caen en la cara interior
THICK  = 0.003  # grosor visual [m]


def aruco2gz(ax, ay, az, ayaw_deg, inset=0.0):
    th     = math.radians(ayaw_deg)
    # Normal de la cara en frame gz = (sin th, -cos th). Empuja `inset` hacia adentro.
    gz_x   = ay + inset * math.sin(th)
    gz_y   = (MAP_W - ax) - inset * math.cos(th)
    gz_z   = az
    yaw_gz = th
    return gz_x, gz_y, gz_z, yaw_gz


def marker_sdf(m):
    mid            = m['id']
    inset          = INSET if str(m.get('group', '')).startswith('wall') else 0.0
    gx, gy, gz, yaw_gz = aruco2gz(m['x'], m['y'], m['z'], m['yaw_deg'], inset)
    # La caja de Gazebo espeja la textura para markers que miran a +gz (yaw 90 o
    # 180). Para esos usamos la textura pre-volteada (<id>_flip.jpg) para que se
    # vea derecha. Ruta portable via GZ_SIM_RESOURCE_PATH (share de description).
    needs_flip     = round(m['yaw_deg']) % 360 in (90, 180)
    tex            = f"{mid}_flip" if needs_flip else f"{mid}"
    img            = f"model://description/textures/arucos/{tex}.jpg"
    # Marcador VERTICAL con imagen derecha: cara (Z local) -> normal horizontal,
    # arriba (Y local) -> +Z mundo.  roll=pi/2, pitch=0, yaw=yaw_gz (=yaw_deg).
    roll           = math.pi / 2
    pitch          = 0.0
    return (
        f'    <model name="aruco_{mid}"><static>true</static>\n'
        f'      <pose>{gx:.4f} {gy:.4f} {gz:.4f} {roll:.5f} {pitch:.5f} {yaw_gz:.5f}</pose>\n'
        f'      <link name="link">\n'
        f'        <visual name="vis">\n'
        f'          <geometry><box><size>{SIDE} {SIDE} {THICK}</size></box></geometry>\n'
        f'          <material>\n'
        f'            <diffuse>1 1 1 1</diffuse>\n'
        f'            <pbr><metal>\n'
        f'              <albedo_map>{img}</albedo_map>\n'
        f'              <metalness>0.0</metalness>\n'
        f'              <roughness>1.0</roughness>\n'
        f'            </metal></pbr>\n'
        f'          </material>\n'
        f'        </visual>\n'
        f'        <collision name="col">\n'
        f'          <geometry><box><size>{SIDE} {SIDE} {THICK}</size></box></geometry>\n'
        f'        </collision>\n'
        f'      </link></model>'
    )


with open(YAML_PATH) as f:
    data = yaml.safe_load(f)

markers = data['markers']

with open(WORLD_PATH) as f:
    content = f.read()

# Eliminar aruco_ models existentes (idempotente)
content = re.sub(
    r'\s*<model name="aruco_\d+".*?</model>',
    '',
    content,
    flags=re.DOTALL
)

# Insertar antes de </world>
blocks = '\n'.join(marker_sdf(m) for m in markers)
content = content.replace('  </world>', blocks + '\n  </world>')

with open(WORLD_PATH, 'w') as f:
    f.write(content)

print(f"OK: {len(markers)} marcadores ArUco insertados en {os.path.basename(WORLD_PATH)}")
for m in markers:
    gx, gy, gz, _ = aruco2gz(m['x'], m['y'], m['z'], m['yaw_deg'])
    print(f"  aruco_{m['id']:2d}  gz=({gx:.3f}, {gy:.3f}, {gz:.3f})  grupo={m['group']}")
