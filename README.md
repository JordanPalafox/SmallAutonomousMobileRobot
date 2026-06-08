<div align="center">

# 🤖 Puzzlebot AMR — Autonomous Forklift Robot

**Autonomous warehouse at scale: in-house SLAM, ArUco localization, A\* navigation, vision, voice and a live digital twin.**

**TE3003B** Challenge · Robotics and Intelligent Systems Integration · Tecnológico de Monterrey (FJ2026)

![ROS 2](https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white)
![Gazebo](https://img.shields.io/badge/Gazebo-Ignition_Fortress-FF6C2C?logo=gazebo&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-MCL-76B900?logo=nvidia&logoColor=white)

</div>

<div align="center">

[![Mission 1 — Puzzlebot AMR](https://img.youtube.com/vi/AEM72Lx7E6k/hqdefault.jpg)](https://www.youtube.com/shorts/AEM72Lx7E6k)

▶️ **[Watch Mission 1 on video (YouTube)](https://www.youtube.com/shorts/AEM72Lx7E6k)**

</div>

---

## 📑 Table of contents
- [About the project](#-about-the-project)
- [Demo](#-demo)
- [Features](#-features)
- [System architecture](#-system-architecture)
- [Tech stack](#-tech-stack)
- [Hardware](#-hardware)
- [Repository structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Most useful commands](#-most-useful-commands)
- [Subsystems in detail](#-subsystems-in-detail)
- [Configuration](#-configuration)
- [Team](#-team)

---

## 🎯 About the project

The **Puzzlebot AMR** is an autonomous forklift robot (Autonomous Mobile Robot) that operates in a
**scaled warehouse (3.65 × 4.85 m)**. It moves pallets between three types of zones:

- 🚚 **Trucks** — loading / unloading
- 🗄️ **Racks** — 2-level shelving
- 🛞 **Rollers** — roller tables (staging)

The entire stack (SLAM, localization, navigation, control, perception, voice) is **built in-house
— no Nav2, no Whisper**. The system runs **distributed**: the on-board Jetson handles real-time
sensing and SLAM, while a laptop runs planning, mission logic, the dashboard and a **digital twin**
that mirrors the real robot live.

> 📷 *Real track and robot:*
> ![Track and robot](assets/pista.jpg)

---

## 🎬 Demo

> 🎥 The **[full Mission 1 (Rollers → Truck)](https://www.youtube.com/shorts/AEM72Lx7E6k)** is in the video above ⬆️

<table align="center">
  <tr>
    <td align="center" width="50%">
      <a href="assets/gemelo.mp4"><img src="assets/gemelo.jpg" height="300" alt="Digital twin in Gazebo"></a>
      <br><b>🪞 Digital twin — Gazebo</b>
      <br><sub>Live mirror of the real robot · click for the video ▶️</sub>
    </td>
    <td align="center" width="50%">
      <img src="assets/dashboard.jpg" height="300" alt="Web dashboard">
      <br><b>🖥️ Web dashboard</b>
      <br><sub>Mission control, telemetry and camera streaming</sub>
    </td>
  </tr>
</table>

---

## ✨ Features

- 🧭 **In-house SLAM in C++** — particle filter (MCL/AMCL) with optional **CUDA** scoring, ICP scan-to-scan + scan-to-map, and **graph SLAM with loop closure**.
- 🎯 **ArUco localization** — 15 fixed beacons that **rescue** the MCL when it drifts in ambiguous areas (they don't replace it; they make it robust).
- 🪞 **Live digital twin** — the virtual robot in Gazebo **follows the real pose** of the physical robot (an environment visualizer, not a physics simulation).
- 🗺️ **Navigation without Nav2** — A\* (global planner) + Bug1 (reactive avoidance) + pure-pursuit.
- 🧠 **YASMIN state machine** — full pick & place missions with a live debugger.
- 👁️ **Vision** — ArUco detection, QR *docking* and logo classifier (vision stop at pick time).
- 🎙️ **Voice without Whisper** — in-house **MFCC → VQ → HMM** pipeline.
- 🖥️ **Web dashboard** — Flask + SocketIO with telemetry, camera streaming and mission control.
- 🏗️ **FPGA-driven lifter** — height control over SPI (Tang Nano 20K).

---

## 🏛️ System architecture

**Distributed** setup (same `ROS_DOMAIN_ID`, clocks synced with chrony):

```
        JETSON NANO (on-board)                         LAPTOP (off-board)
  ┌───────────────────────────────┐           ┌──────────────────────────────────┐
  │ robot.launch.py               │           │ laptop.launch.py                  │
  │  • LiDAR /scan  • micro-ROS    │   WiFi    │  • navigation (A* + Bug1)         │
  │  • real_odom → /…/odom         │ ◄═══════► │  • mission_control (YASMIN)       │
  │  • slam_node (MCL+CUDA+ICP)    │   DDS     │  • dashboard (Flask)              │
  │    → /slam_pose, /map, TF      │           │  • voice_control (MFCC+VQ+HMM)    │
  │  • aruco_localization          │           │  • RViz + digital twin (Gazebo)   │
  │  • lifting (FPGA/SPI)          │           │    └ gemelo_mirror ← /slam_pose   │
  │  • qr docking / logo stop      │           └──────────────────────────────────┘
  └───────────────────────────────┘
```

**Control chain (`/cmd_vel`)** — *every* upstream node publishes to **`/cmd_vel_in`**:
```
/cmd_vel_in ──► twist_relay (anti-brownout ramp) ──► /cmd_vel ──► micro-ROS ──► motors
```

**Localization (hierarchy):** Monte Carlo (LiDAR vs map) is the **primary** localizer; the
ArUcos are **rescue beacons** that re-anchor the belief when it drifts.
```
camera /video_source/raw ─► aruco_localization ─► /aruco_pose_estimate ─► slam_node.onAruco
                                                                          (re-seeds the MCL)
```

---

## 🛠️ Tech stack

| Layer | Technology |
|---|---|
| Middleware | **ROS 2 Humble** (Ubuntu 22.04) |
| Simulation / twin | **Gazebo Ignition Fortress** + `ros_gz` |
| SLAM | In-house C++: **MCL/AMCL** (particles) · **CUDA** (scoring) · **ICP** scan-match · **pose-graph** + loop closure · `Eigen` |
| Localization | **ArUco** `DICT_ARUCO_ORIGINAL` (5×5, 9 cm) — MCL rescue beacons |
| Navigation | **A\*** + **Bug1** + **pure-pursuit** (no Nav2) |
| Missions | **YASMIN** (state machine) with a live debugger |
| Vision | **OpenCV** — ArUco, QR pose (`solvePnP`/IPPE), logo classifier (`weights.pt`) |
| Voice | In-house **MFCC + VQ + HMM** (`numpy`/`scipy`, no Whisper) |
| Dashboard | **Flask + Flask-SocketIO** + `cv2` streaming |
| Control | `ros2_control` (Gazebo only) · in-house differential kinematics (real) |
| Lifter | **FPGA Tang Nano 20K** over **SPI** (`/dev/spidev0.0`) |

---

## 🔩 Hardware

| Component | Detail |
|---|---|
| On-board compute | **NVIDIA Jetson Nano** |
| LiDAR | **RPLidar A1** → `/scan` (10 Hz) |
| Camera | **PiCamera** (640×360) → `/video_source/raw` |
| Encoders / motors | Hackerboard MCR2 via **micro-ROS** (`/VelocityEncL,R`, `/cmd_vel`) |
| Lifter | **FPGA Tang Nano 20K** (SPI) → height levels |
| Markers | 15 **ArUco** 5×5, **9 cm**, fixed on walls and rack sides |
| Track | Scaled warehouse **3.65 × 4.85 m** |

---

## 📂 Repository structure

```
src/
├── bringup/          # Orchestration launches (sim, robot, laptop, all_pc)
├── description/      # URDF/xacro, Gazebo worlds (almacen_racks), meshes, twin
├── controller/       # Odometry, kinematics, twist_relay, gemelo_mirror
├── slam/             # C++: slam_node (MCL+CUDA+ICP+graph), puzzlebot_sim
├── navigation/       # A* + Bug1 + pure-pursuit, waypoint_recorder
├── perception/       # ArUco, QR docking, logo classifier, calibration
├── mission_control/  # YASMIN state machine (missions)
├── voice_control/    # MFCC + VQ + HMM
├── dashboard/        # Flask + SocketIO web
└── lifting/          # GPIO/SPI HAL → FPGA
```

---

## ⚙️ Installation

**Requirements:** Ubuntu 22.04 + ROS 2 Humble.

```bash
# 1. Clone inside a workspace
git clone https://github.com/JordanPalafox/SmallAutonomousMobileRobot.git
cd SmallAutonomousMobileRobot

# 2. System dependencies (required for the sim/twin!)
sudo apt update && sudo apt install -y \
  ros-humble-joint-state-publisher ros-humble-joint-state-publisher-gui \
  ros-humble-topic-tools ros-humble-ros-gz \
  ros-humble-robot-state-publisher ros-humble-xacro ros-humble-rviz2

# 3. Python dependencies
pip install numpy scipy opencv-contrib-python yasmin flask flask-socketio

# 4. Build and source
colcon build --symlink-install
source install/setup.bash
```

---

## ▶️ Usage

### Full simulation (one command)
Gazebo (twin) + SLAM + navigation + missions + dashboard + RViz:
```bash
ros2 launch bringup sim.launch.py
ros2 launch bringup sim.launch.py start_mode:=mapping   # build the map first
```

### Real robot (distributed)
```bash
# On the Jetson:
ros2 launch bringup robot.launch.py

# On the laptop (live digital twin with gemelo:=true):
ros2 launch bringup laptop.launch.py gemelo:=true
```

### Real robot on a single machine
```bash
ros2 launch bringup all_pc.launch.py
```

---

## ⌨️ Most useful commands

> Remember to **`source install/setup.bash`** in every new terminal.

### 🔨 Build & environment
```bash
colcon build --symlink-install               # build everything
colcon build --packages-select slam          # rebuild a single package
source install/setup.bash                     # source the workspace
```

### 🚀 Launch the system
```bash
ros2 launch bringup sim.launch.py                      # full SIM (twin+SLAM+nav+missions+dashboard)
ros2 launch bringup sim.launch.py start_mode:=mapping  # build the map first
ros2 launch bringup robot.launch.py                    # [Jetson] real robot (sensing+SLAM+lifter)
ros2 launch bringup laptop.launch.py gemelo:=true      # [laptop] nav+missions+dashboard+RViz+live twin
ros2 launch bringup all_pc.launch.py                   # everything on a single machine
```

### 🎮 Move the robot (everything goes through `/cmd_vel_in`)
```bash
ros2 topic pub /cmd_vel_in geometry_msgs/msg/Twist "{linear: {x: 0.15}, angular: {z: 0.0}}" -r 10
ros2 launch controller joystick_teleop.launch.py       # joystick teleop
```

### 🧠 Missions (easiest: the dashboard at http://localhost:8080)
```bash
ros2 topic echo /robot_state                                                   # current SM state
ros2 topic pub --once /mission std_msgs/msg/String "{data: '{\"type\":\"rollers\"}'}"  # Mission 1: Rollers → Truck
ros2 topic pub --once /mission std_msgs/msg/String "{data: '{\"type\":\"racks\"}'}"    # Mission 2: Racks → Truck
ros2 topic pub --once /sm/control std_msgs/msg/String "{data: '{\"action\":\"abort\"}'}"  # abort | pause | resume | step
```

### 🗺️ SLAM & map
```bash
ros2 topic echo /slam_pose                             # estimated pose (the one the twin follows)
ros2 service call /map_saver/save_map std_srvs/srv/Trigger   # save map to ~/ros2_maps/
```

### 🎯 ArUco
```bash
ros2 topic echo /aruco_ids                             # IDs detected right now
ros2 topic echo /aruco_pose_estimate                   # rescue pose published to the MCL
rqt_image_view /aruco_localization/debug_image         # see the detection live
python3 src/perception/tools/aruco_map_editor.py       # visual map editor (http://127.0.0.1:8770)
```

### 🏗️ Lifter
```bash
ros2 topic pub --once /lifter_level std_msgs/msg/UInt8 "{data: 3}"   # raise to level 3 (0–5)
```

### 🪞 Digital twin (Gazebo)
```bash
python3 scripts/gen_warehouse.py                       # regenerate world racks/rollers
python3 scripts/add_arucos_to_world.py                 # regenerate world ArUcos
ign gazebo -g                                          # open the Gazebo GUI connected to the server
```

### 🔍 Diagnostics
```bash
ros2 topic hz /scan                                    # is the LiDAR coming in at ~10 Hz?
ros2 node list ; ros2 topic list                       # what's running / what's being published?
pkill -9 -f "ign gazebo" ; pkill -x Xvfb               # clean up hung Gazebo/Xvfb
```

---

## 🧩 Subsystems in detail

### 🧭 SLAM (`slam`)
`slam_node` (C++) runs on every `/scan` (10 Hz): **predict** (odom + ICP scan-to-scan as a
slip-robust rotation prior) → **update** (likelihood-field sensor model against the map, with
optional **CUDA** scoring) → **resample** → **scan-to-map refine**. A back-end thread does
**graph SLAM** (keyframes + loop closure) and re-rasterizes the map. Modes `mapping` / `navigation`
(loads a saved `.pgm`). `puzzlebot_sim` is an in-house 2D simulator that models slip and LiDAR
latency to tune SLAM realistically.

> 🗺️ *RViz: SLAM map + MCL particle cloud (arrows) + LiDAR scan + detected ArUcos with their IDs (top, the `debug_image` from `aruco_localization`):*
> ![SLAM Monte Carlo + ArUco](assets/mapa_montecarlo.jpg)

### 🎯 ArUco localization (`perception` + `slam`)
`aruco_localization` detects the markers, computes the robot's absolute pose
(`T_map_base = T_map_marker · inv(T_cam_marker) · inv(T_base_cam)`) and publishes `/aruco_pose_estimate`.
`slam_node` uses it as a **rescue beacon**: if MCL disagrees by more than `aruco_snap_tol`, it re-seeds
the belief there and the laser snaps it to the walls. Map in `perception/config/aruco_map.yaml`.
Visual editor included: `python3 src/perception/tools/aruco_map_editor.py`.

### 🪞 Digital twin (`controller/gemelo_mirror` + `description`)
The `almacen_racks.world` world reproduces the real track (walls, racks, rollers and textured ArUcos).
`gemelo_mirror` subscribes to `/slam_pose` and **teleports** the virtual robot (static, no physics)
via Ignition's `set_pose` service → a **live mirror** of the real robot to understand its environment.
Rack/roller layout is reproducible from `config/rack_groups.yaml` with `scripts/gen_warehouse.py`.

### 🗺️ Navigation (`navigation`)
`nav_node` loads the map + `waypoints.yaml`, receives waypoint names (`truck_*`, `rack_*`, `roller_*`),
plans with **A\*** and follows the path with **pure-pursuit**; avoids with **Bug1** (wall-follow) and
re-plans. Publishes to `/cmd_vel_in`.

### 🧠 Missions (`mission_control`)
**YASMIN** state machine: `search → nav_to_truck → pick / pick_from_rack → place / release_load`.
It receives missions via `/mission` (JSON, from voice or dashboard), commands nav and lifter, and
exposes a live **debugger** (`/sm/control`: pause, step, force_outcome, abort).

### 👁️ Vision (`perception`)
ArUco (localization), **QR docking** (`qr_quad_alignment`, pixel-space alignment + pick maneuver),
and the **E80 logo classifier** (`logo_stop_debug` + `weights.pt`) for the vision stop before bumping
into the load.

### 🎙️ Voice (`voice_control`)
In-house pipeline with no pre-trained models: `audio → VAD → MFCC → VQ (codebook K=256) → HMM →
grammar → mission`. Publishes `/voice_command` and `/mission`.

### 🖥️ Dashboard (`dashboard`)
**Flask + SocketIO** server (port 8080): telemetry, camera streaming, mission status, waypoint
capture, map saving and mode switching.

### 🏗️ Lifter (`lifting`)
`lifting_node` translates `/lifter_level` (UInt8) to the **FPGA Tang Nano 20K** over **SPI**; the FPGA
maps each level to a servo angle. Includes a mock HAL for simulation.

---

## 🔧 Configuration

| File | Purpose |
|---|---|
| `perception/config/aruco_map.yaml` | Real ArUco positions (source of truth) |
| `perception/config/camera_params.yaml` | Camera intrinsics (640×360) |
| `slam/config/slam_params.yaml` | MCL tuning (sigma_hit, alphas, aruco_*, etc.) |
| `navigation/config/waypoints.yaml` | Named waypoints (zones) |
| `description/config/rack_groups.yaml` · `roller_groups.yaml` | Twin layout (→ `gen_warehouse.py`) |
| `~/ros2_maps/warehouse.yaml` | Saved wall map (navigation mode) |

---

## 👥 Team

**TE3003B** Challenge project — Robotics and Intelligent Systems Integration,
Tecnológico de Monterrey (FJ2026).

| Member | ID |
|---|---|
| Victor Alejandro Meneses | A01384002 |
| Juan José Jáuregui Barba | A00836722 |
| Hugo Daniel Castillo Ovando | A00836025 |
| Rosendo De Los Ríos Moreno | A01198515 |
| Jordan Arturo Palafox Salinas | A00835705 |

---

<div align="center">
Built with ROS 2 · no Nav2 · no Whisper — everything in-house.
</div>
