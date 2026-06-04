# 3.16 works for the CPU-only build (Jetson Nano ships CMake 3.16).  The CUDA
# path (gated by SLAM_USE_CUDA, dev box only) needs >=3.18 for CUDA_ARCHITECTURES,
# but dev boxes run a newer cmake anyway, so the minimum can stay at 3.16.
cmake_minimum_required(VERSION 3.16)
project(slam LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

set(SLAM_DEPS rclcpp sensor_msgs nav_msgs geometry_msgs std_msgs std_srvs
              tf2 tf2_ros tf2_geometry_msgs visualization_msgs)

# ── Optional CUDA ────────────────────────────────────────────────
# SLAM_USE_CUDA=OFF forces the CPU stub (no nvcc).  Needed on the Jetson Nano
# (CUDA 10.2's nvcc cannot compile the C++17 host code, and a CUDA build can
# OOM the 2 GB board).  CPU AMCL fits the 10 Hz / 100 ms budget on the A57s, so
# build the robot with:  colcon build --packages-select slam --cmake-args -DSLAM_USE_CUDA=OFF
option(SLAM_USE_CUDA "Build the CUDA GPU MCL (OFF → CPU stub)" ON)
include(CheckLanguage)
set(SLAM_HAS_CUDA FALSE)
if(SLAM_USE_CUDA)
  check_language(CUDA)
endif()
if(SLAM_USE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(SLAM_HAS_CUDA TRUE)
  # Target both the deployment GPU and dev GPUs:
  #   53 = Jetson Nano (Maxwell, sm_53)  ← the actual robot
  #   62 = Jetson TX2   (Pascal)
  #   72 = Xavier/AGX   (Volta)
  #   86 = RTX 30-series laptop (dev box)
  # Override at configure time with -DCMAKE_CUDA_ARCHITECTURES=... .
  if(NOT DEFINED SLAM_CUDA_ARCHS)
    set(SLAM_CUDA_ARCHS "53;62;72;86")
  endif()
  message(STATUS "slam: CUDA found (${CMAKE_CUDA_COMPILER}) — building GPU MCL for arch ${SLAM_CUDA_ARCHS}")
else()
  message(STATUS "slam: no CUDA — building CPU-only MCL")
endif()

# ── Core SLAM library ────────────────────────────────────────────
set(CORE_SRC src/occupancy_grid.cpp src/icp.cpp src/amcl.cpp src/pose_graph.cpp)
if(SLAM_HAS_CUDA)
  list(APPEND CORE_SRC src/cuda_mcl.cu)
else()
  list(APPEND CORE_SRC src/cuda_stub.cpp)
endif()

add_library(slam_core ${CORE_SRC})
target_include_directories(slam_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(slam_core PUBLIC Eigen3::Eigen)
if(SLAM_HAS_CUDA)
  set_target_properties(slam_core PROPERTIES CUDA_ARCHITECTURES "${SLAM_CUDA_ARCHS}")
  target_compile_definitions(slam_core PRIVATE SLAM_HAS_CUDA=1)
endif()

# ── slam_node executable ─────────────────────────────────────────
add_executable(slam_node src/slam_node.cpp)
target_link_libraries(slam_node slam_core)
ament_target_dependencies(slam_node ${SLAM_DEPS})

# ── puzzlebot_sim executable ─────────────────────────────────────
add_executable(puzzlebot_sim src/puzzlebot_sim.cpp)
ament_target_dependencies(puzzlebot_sim ${SLAM_DEPS})

install(TARGETS slam_node puzzlebot_sim
  DESTINATION lib/${PROJECT_NAME})

# Python helper node (map saver) — installed as `ros2 run slam map_saver`.
install(PROGRAMS scripts/map_saver_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME map_saver)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

ament_package()
