FROM osrf/ros:humble-desktop-full

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-bridge \
    ros-humble-ign-ros2-control \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-joy \
    ros-humble-joy-teleop \
    ros-humble-twist-mux \
    ros-humble-slam-toolbox \
    ros-humble-teleop-twist-keyboard \
    ros-humble-rviz2 \
    ros-humble-tf2-ros \
    ros-humble-tf-transformations \
    ros-humble-nav-msgs \
    ros-humble-sensor-msgs \
    ros-humble-topic-tools \
    xterm \
    tmux \
    micro \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install flask-socketio

WORKDIR /puzzlebot_ws

COPY src/ src/

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install"

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'source /puzzlebot_ws/install/setup.bash' >> /root/.bashrc

COPY .devcontainer/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["bash"]
