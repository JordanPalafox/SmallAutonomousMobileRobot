#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /puzzlebot_ws/install/setup.bash

exec "$@"
