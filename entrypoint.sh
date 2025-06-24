#!/bin/bash
set -e

# Setup environment ROS setiap kali kontainer dimulai
source "/opt/ros/noetic/setup.bash"
export GAZEBO_RESOURCE_PATH=/root/catkin_ws/src/multi_robot_scenario/launch
source "/root/catkin_ws/devel/setup.bash"


# Jalankan perintah apa pun yang diberikan saat 'docker run' (misalnya, 'python3 train_parallel.py')

xterm -hold -e "roslaunch pclprocess train.launch rviz:=false" &

sleep 30
xterm -hold -e "python3 ~/drl_ws/train_parallel.py" &
exec "$@"
