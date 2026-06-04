# source ~/lifting_env.sh  -> deja la terminal lista para hablar con el lifting_node
source /opt/ros/humble/setup.bash
source ~/ros2_packages_ws/install/local_setup.bash 2>/dev/null
source ~/ros2_ws/install/local_setup.bash 2>/dev/null
source ~/puzzlebot_challenge_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/.ros/fastrtps_local.xml
echo "env lifting listo: DOMAIN=$ROS_DOMAIN_ID  PROFILE=$FASTRTPS_DEFAULT_PROFILES_FILE"
