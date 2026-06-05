# Localización por marcadores ArUco — Puzzlebot AMR

Capa de localización por balizas **ArUco** que hace más robusto al SLAM (Monte
Carlo) del Puzzlebot. Cubre el **robot real**; el gemelo de Gazebo (visualización)
se documenta aparte.

---

## 1. Filosofía

- **Monte Carlo (AMCL + LiDAR vs mapa) es el localizador PRINCIPAL.** Corre
  continuo y hace el trabajo fino contra las paredes.
- **Los ArUcos son balizas de RESCATE.** No localizan momento a momento; solo
  **re-anclan** al MCL cuando deriva o se pierde (zonas ambiguas, pasillos
  simétricos, pocas paredes). Hacen a MCL más robusto, **no lo reemplazan**.

---

## 2. Pipeline

```
┌─ CÁMARA ──────────────────────────────────────────────────────────┐
│ PiCamera frontal (16 cm) → /video_source/raw  (sensor_msgs/Image) │
└───────────────────────────────┬───────────────────────────────────┘
                                 ▼
┌─ aruco_localization  (paquete perception, corre en la Jetson) ────┐
│ 1. Detecta ArUcos (DICT_ARUCO_ORIGINAL 5×5, 9 cm) + cv2.solvePnP   │
│    → pose del marcador en frame de cámara (T_cam_marker)           │
│ 2. Busca el ID en aruco_map.yaml → su pose en el mundo            │
│    (T_map_marker)                                                  │
│ 3. T_map_base = T_map_marker · inv(T_cam_marker) · inv(T_base_cam) │
│    → pose del ROBOT en el mapa                                     │
│ 4. Aplica offset esquina→centro (aruco_origin_in_map)             │
│ 5. Si ve varios marcadores, fusiona (media ponderada por 1/dist²) │
└───────────────────────────────┬───────────────────────────────────┘
                                 ▼
         /aruco_pose_estimate  (geometry_msgs/PoseWithCovarianceStamped, map)
                                 ▼
┌─ slam_node.onAruco  (paquete slam, MCL) ──────────────────────────┐
│ ¿La pose del ArUco coincide con la creencia del MCL?             │
│   · SÍ (dentro de aruco_snap_tol)  → no hace nada (deja al láser) │
│   · NO (derivó)  → REPOSICIONA la creencia ahí (initGaussian) y   │
│     el siguiente scan encaja la pose contra las paredes           │
└────────────────────────────────────────────────────────────────────┘
```

**Topics que publica `aruco_localization`:**

| Topic | Tipo | Descripción |
|---|---|---|
| `/aruco_pose_estimate` | `PoseWithCovarianceStamped` | Pose del robot en `map` (la consume el SLAM) |
| `/aruco_poses` | `PoseArray` | Poses de marcadores en frame de cámara |
| `/aruco_ids` | `Int32MultiArray` | IDs detectados (mismo orden que `/aruco_poses`) |
| `/aruco_localization/debug_image` | `Image` | Imagen anotada con las detecciones |

---

## 3. Geometría

Cada ArUco es un punto fijo con ID único cuya posición **conoces** (la mediste).
Al verlo, `solvePnP` da **dónde está el marcador respecto a la cámara**. Como
sabes dónde está el marcador en el mundo (`aruco_map.yaml`) y dónde está la
cámara en el robot (`cam_xyz`, `cam_pitch_deg`), despejas **dónde está el robot
en el mundo**. Es trilateración con un solo marcador.

**Frames y offset clave:** los marcadores se miden desde la **esquina** `(0,0)`
de la pista, pero el SLAM pone su origen donde **arranca el robot** (el
**centro**). El parámetro `aruco_origin_in_map: [-1.825, -2.425, 0]`
(= −mitad de 3.65 × 4.85) convierte coords-esquina al frame del SLAM. Así puedes
seguir midiendo cómodo desde la esquina.

Convención del mapa (REP-103): `x` adelante, `y` izquierda, `z` arriba.
`yaw_deg` = dirección a la que MIRA la cara del marcador (0=+X, 90=+Y, 180=−X,
270=−Y). `z` = altura del centro del marcador.

---

## 4. Archivos clave

| Archivo | Qué es |
|---|---|
| `src/perception/perception/aruco_detector.py` | Detector (DICT_ARUCO_ORIGINAL, solvePnP); compatible OpenCV 4.5.x (Jetson) y 4.7+ |
| `src/perception/perception/aruco_localization.py` | Nodo: detecta → estima pose del robot → `/aruco_pose_estimate` |
| `src/perception/config/aruco_map.yaml` | Posiciones REALES de los marcadores (id, x, y, z, yaw_deg) |
| `src/perception/config/camera_params.yaml` | Intrínsecos de la cámara (320×240, marker 9 cm) |
| `src/perception/tools/aruco_map_editor.py` | GUI web para colocar los marcadores sobre el plano y escribir `aruco_map.yaml` |
| `src/slam/src/slam_node.cpp` (`onAruco`) | Integración con el MCL (baliza de rescate) |
| `src/bringup/launch/robot.launch.py` | Lanza `aruco_localization` en la Jetson |

---

## 5. Comandos

### Compilar
```bash
cd ~/Projects/puzzlebot/SmallAutonomousMobileRobot
colcon build --packages-select perception slam bringup
source install/setup.bash
```

### Lanzar (robot real, en la Jetson)
```bash
ros2 launch bringup robot.launch.py
# incluye: controller + slam_node + aruco_localization + lifter + qr + logo
```
Argumentos útiles:
```bash
ros2 launch bringup robot.launch.py aruco_cam_pitch_deg:=10   # tilt cámara abajo
ros2 launch bringup robot.launch.py aruco:=false              # desactivar arucos
```

### Verificar
```bash
ros2 topic echo /aruco_ids                 # ¿qué IDs ve? (vacío = no detecta)
ros2 topic echo /aruco_pose_estimate       # pose del robot en `map`
ros2 run rqt_image_view rqt_image_view /aruco_localization/debug_image
ros2 topic echo /slam_pose                 # pose final del MCL
```

### Editar el mapa de marcadores (GUI)
```bash
python3 src/perception/tools/aruco_map_editor.py   # abre http://127.0.0.1:8770
```

---

## 6. Parámetros que más se tocan

**Cámara** (`robot.launch.py` o al lanzar):

| Param | Default | Qué es |
|---|---|---|
| `cam_xyz` | `[0.10, 0, 0.16]` | posición de la cámara en el robot (16 cm del piso) |
| `aruco_cam_pitch_deg` | `0` | inclinación hacia abajo; súbelo si pierde marcadores bajos |
| `aruco_origin_in_map` | `[-1.825, -2.425, 0]` | offset esquina→centro de la pista |
| `marker_length` | `0.09` | lado del marcador (9 cm) |
| `max_range` | `2.0` | descarta marcadores más lejanos [m] |

**Rescate MCL** (`src/slam/config/slam_params.yaml`):

| Param | Default | Qué es |
|---|---|---|
| `aruco_enabled` | `true` | activar/desactivar la baliza |
| `aruco_snap_tol` | `0.15` m | cuánto debe derivar el MCL para que el ArUco intervenga |
| `aruco_snap_tol_theta` | `0.15` rad | ídem, angular |
| `aruco_seed_sigma_xy` | `0.20` m | dispersión al re-anclar (deja que el láser afine) |
| `aruco_global_init` | `false` | si `true`, el 1er ArUco fija la pose (colocar el robot a ciegas) |

---

## 7. Pendientes de calibración (físico, no código)

1. **Tilt real de la cámara** → ajusta `aruco_cam_pitch_deg`. Los marcadores
   quedaron bajos (~4–8 cm) y la cámara está a 16 cm, así que quedan por debajo;
   un leve tilt hacia abajo ayuda.
2. **El mapa de paredes** (`.pgm`/`.yaml`) debe haberse hecho con el robot
   arrancando en el **centro** (para que el offset `aruco_origin_in_map` cuadre).
   Si no, re-mapear desde el centro.
3. **Posiciones finales** de los marcadores en `aruco_map.yaml` (vía la GUI).

---

## 8. Marcadores

- Diccionario: **DICT_ARUCO_ORIGINAL** (rejilla 5×5).
- Tamaño: **9 cm** por lado.
- IDs (15 en uso): 5, 7, 13, 17, 19, 20, 21, 22, 25, 28, 30, 31, 37, 38, 41.
- Montaje: **verticales** en muros / costados de obstáculos, mirando a la pista.
- PDF para imprimir: `arucos/data/markers_aruco5x5/marcadores_ARUCO5x5_9cm_rejilla.pdf`.

---

## 9. Dependencias del sistema

Además del `colcon build`, en la máquina hacen falta estos paquetes apt:
```bash
sudo apt install -y ros-humble-joint-state-publisher \
                    ros-humble-joint-state-publisher-gui \
                    ros-humble-topic-tools
```

---

## 10. Troubleshooting

| Síntoma | Causa / fix |
|---|---|
| `/aruco_ids` vacío | No detecta: marcador fuera de FOV (sube `aruco_cam_pitch_deg`), poca luz, o resolución/enfoque |
| Pose corrida | Revisa `cam_xyz`/`cam_pitch_deg` reales y que el `.pgm` se haya mapeado desde el centro |
| El ArUco rescata muy seguido | Sube `aruco_snap_tol` (deja más libre a MCL) |
| El ArUco no rescata nunca | Baja `aruco_snap_tol`, o verifica que llega `/aruco_pose_estimate` |
