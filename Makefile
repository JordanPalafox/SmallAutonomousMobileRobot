SHELL := /bin/bash
ROS_SETUP := /opt/ros/humble/setup.bash
WS := $(shell pwd)

.PHONY: build clean rebuild install test

build:
	@source $(ROS_SETUP) && colcon build --symlink-install

rebuild: clean build

clean:
	@rm -rf build install log

install:
	@source $(ROS_SETUP) && source install/local_setup.bash

test:
	@source $(ROS_SETUP) && source install/local_setup.bash && colcon test --return-code-on-test-failure
