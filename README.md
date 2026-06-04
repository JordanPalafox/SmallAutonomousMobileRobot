# Puzzlebot-Challenge-
This repository is for the implementation of the class "Integración de robótica y sistemas inteligentes" at Tecnologico de Monterrey

## Week 1 — Robot Visualization in RViz

Spawns the MCR2 Puzzlebot (Jetson + Lidar edition) in RViz. The robot automatically moves in a circle around the origin while the wheels spin.

### Requirements

#### Option 1: Docker (Recommended)

Build the Docker image:

```bash
docker build -t puzzlebot -f .devcontainer/Dockerfile .
```

Run the container:

```bash
docker run -it --rm --gpus all --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  puzzlebot bash
```

#### Option 2: System dependencies (ROS 2 Humble on Ubuntu 22.04)

Core ROS 2 tools and robot description utilities:

```bash
sudo apt-get install -y \
  ros-humble-robot-state-publisher \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui \
  ros-humble-rviz2 \
  ros-humble-xacro \
  ros-humble-tf2-ros \
  ros-humble-teleop-twist-keyboard
```

Gazebo (Ignition Fortress) + ROS 2 bridge:

```bash
sudo apt-get install -y \
  ros-humble-ros-gz-sim \
  ros-humble-ros-gz-bridge \
  ros-humble-ros-gz-interfaces
```

ros2_control and controllers:

```bash
sudo apt-get install -y \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-controller-manager \
  ros-humble-ign-ros2-control
```

Joystick / teleoperation / mux:

```bash
sudo apt-get install -y \
  ros-humble-joy \
  ros-humble-joy-teleop \
  ros-humble-twist-mux
```

<!-- SLAM toolbox (used by `puzzlebot_mapping`):

```bash
sudo apt-get install -y \
  ros-humble-slam-toolbox
``` -->

Python dependencies:

```bash
pip install numpy scipy
```

#### Build and source the workspace

```bash
git clone https://github.com/Hugo734/Puzzlebot-Challenge-.git
cd Puzzlebot-Challenge-
colcon build --packages-select puzzlebot_description
source install/setup.bash
```

### Demo

<img src="assets/img.png" width="49%"/> <img src="assets/img1.png" width="49%"/>

### Launch

```bash
ros2 launch puzzlebot_description week1.launch.py
```

### What it runs

| Node | Description |
|---|---|
| `robot_state_publisher` | Loads the URDF and publishes the robot TF tree |
| `circular_motion.py` | Moves the robot in a circle (radius 0.5 m) and spins the wheels |
| `rviz2` | Visualizes the robot with `odom` as the fixed frame |
