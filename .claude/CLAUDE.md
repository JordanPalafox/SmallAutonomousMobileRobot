# Puzzlebot AMR — CLAUDE.md

## Proyecto

Robot montacargas autónomo (AMR) basado en el **Puzzlebot** para el reto TE3003B (ITESM FJ2026).
El robot mueve pallets entre tres zonas de un almacén a escala:
- **Camiones** (3 unidades) — zona de carga/descarga
- **Racks** (varios, 2 niveles de altura) — estanterías de almacenamiento
- **Rollers** (varios) — mesas de rodillos como staging area

No usar Nav2. Toda la navegación, SLAM y control es implementación propia.

---

## Hardware

| Componente | Detalle |
|---|---|
| Computadora | Jetson Nano 4GB |
| GPIO library | `Jetson.GPIO` |
| FPGA lifter | 3 pines GPIO → 8 niveles de altura (encoding binario 000–111) |
| Sensores | LiDAR (`/scan`), Cámara (`/cam_img`), Encoders (`/VelocityEncR`, `/VelocityEncL`) |
| Actuadores | Motores DC vía H-bridge (`/cmd_vel`) |
| Detección pallets | Solo color/forma (sin QR ni Aruco en pallets) |

### Encoding del lifter (3 bits → GPIO Jetson → FPGA)

```
000 = Nivel 0 (piso / transporte)
001 = Nivel 1
010 = Nivel 2
011 = Nivel 3 (pick pallet inferior)
100 = Nivel 4
101 = Nivel 5 (pick pallet rack nivel 2)
110 = Nivel 6
111 = Nivel 7 (altura máxima)
```

---

## Paquetes existentes (renombrar — quitar prefijo `puzzlebot_`)

| Nombre actual (en disco) | Nombre objetivo | Lenguaje | Propósito |
|---|---|---|---|
| `puzzlebot_bringup` | `bringup` | CMake | Launch files del robot |
| `puzzlebot_controller` | `controller` | Python | Cinemática diferencial, odometría, TF |
| `puzzlebot_description` | `description` | CMake/URDF | Modelo URDF + mundos Gazebo |
| `puzzlebot_localization_cpp` | `localization` | C++ | EKF + ICP 2D para localización |
| `puzzlebot_slam` | `slam` | Python | SLAM con ICP + OccupancyGrid |

---

## Paquetes a crear

| Paquete | Lenguaje | Propósito |
|---|---|---|
| `navigation` | Python | A* + Bug1 + PID seguidor de trayectoria |
| `mission_control` | Python (YASMIN) | Máquina de estados del AMR |
| `perception` | Python | Aruco, CNN tráiler, alineación PID con pallet |
| `lifting` | Python | HAL GPIO → FPGA (Jetson.GPIO, 3 bits) + mock para sim |
| `voice_control` | Python | Reconocimiento de voz LPC + VQ, parser de comandos |
| `dashboard` | Python/Flask | Interfaz web con telemetría, streaming y estado de misiones |

---

## Interfaces ROS2 principales

| Topic | Tipo | Dirección | Descripción |
|---|---|---|---|
| `/cmd_vel` | `Twist` | → robot | Velocidad lineal y angular |
| `/scan` | `LaserScan` | ← LiDAR | Escaneo láser |
| `/cam_img` | `Image` | ← cámara | Imagen cruda |
| `/robot_pose` | `PoseStamped` | ← EKF | Pose estimada del robot |
| `/map` | `OccupancyGrid` | ← SLAM | Mapa 2D del entorno |
| `/mission` | `String` (JSON) | ← web/voz | Misión actual `{source, dest, pallet_id}` |
| `/robot_state` | `String` | publicado por SM | Estado actual de la máquina de estados |
| `/lifter_level` | `UInt8` | → lifter node | Nivel 0–7 del lifter |
| `/alignment_error` | `Point` | ← visión | Error de alineación en píxeles (x, y) |
| `/voice_command` | `String` | ← voice node | Comando reconocido |
| `/aruco_poses` | `PoseArray` | ← visión | Poses de markers Aruco detectados |
| `/trailer_detection` | `BoundingBox2D` | ← visión | Detección de tráiler con CNN |

---

## Máquina de estados (YASMIN)

Librería: **YASMIN** (`pip install yasmin`) — diseñada para ROS2 Python, tiene visualizador propio.

```
WAITING_COMMAND
    ↓ [mission_received]
NAVIGATING_TO_SOURCE        ← A* + Bug algorithm
    ↓ [arrived]
ALIGNING_TO_PALLET          ← PID con visión color/forma
    ↓ [aligned]
PICKING_LOWER | PICKING_UPPER | PICKING_FROM_TRUCK   ← lifter + movimiento
    ↓ [picked]
NAVIGATING_TO_DEST          ← A* + Bug algorithm
    ↓ [arrived]
ALIGNING_TO_DEST            ← PID con visión
    ↓ [aligned]
PLACING_PALLET              ← lifter baja
    ↓ [placed]
WAITING_COMMAND

Interrupciones: PAUSED (voz/web), EMERGENCY_STOP
```

---

## Navegación

- Mapa: **pre-guardado** (`.pgm` + `.yaml`), SLAM activo solo para correcciones de localización
- Zonas: **posiciones fijas** definidas en `config/waypoints.yaml`
- Planificador global: **A\***
- Evasión reactiva: **Bug1 / Tangent Bug**
- Seguimiento de trayectoria: **PID** (velocidad angular basada en error de heading)

---

## Reconocimiento de voz

Sin Whisper ni modelos pre-entrenados. Pipeline propio:

```
Audio (16kHz) → VAD (energía adaptativa) → LPC orden 12 → LSF → VQ codebooks
→ secuencia de palabras → parser de gramática → misión ROS2
```

Vocabulario objetivo: `start, stop, pause, next, ve, rack, camion, roller, nivel, uno, dos, recoge, deja`

---

## Decisiones de diseño

- **No Nav2**: toda la pila de navegación es implementación propia
- **No Whisper**: reconocimiento de voz con LPC + VQ (inspirado en JordanPalafox/Practica-1)
- **YASMIN** sobre SMACH: mejor soporte ROS2, más mantenida, visualizador incluido
- **A\*** sobre RRT/D\*: mapa conocido y estático, A* es suficiente y más simple
- **Bug1** sobre Bug0: más robusto ante obstáculos cóncavos
- **Jetson.GPIO** para control FPGA: nativo para Jetson Nano
- **Flask** para la web: ligero, suficiente para streaming + REST + WebSocket
