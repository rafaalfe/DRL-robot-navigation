#!/bin/bash
set -e

# Setup environment ROS setiap kali kontainer dimulai
source "/opt/ros/noetic/setup.bash"
export GAZEBO_RESOURCE_PATH=/root/catkin_ws/src/multi_robot_scenario/launch
source "/root/catkin_ws/devel/setup.bash"


# Jalankan perintah apa pun yang diberikan saat 'docker run' (misalnya, 'python3 train_parallel.py')

xterm -hold -e "export ROS_MASTER_URI=http://localhost:11311; export ROS_HOSTNAME=localhost; roslaunch pclprocess train.launch rviz:=false port:=11346" &

sleep 10

xterm -hold -e "export ROS_MASTER_URI=http://localhost:11312; export ROS_HOSTNAME=localhost; roslaunch pclprocess train.launch rviz:=false port:=11347" &

sleep 10

xterm -hold -e "export ROS_MASTER_URI=http://localhost:11313; export ROS_HOSTNAME=localhost; roslaunch pclprocess train.launch rviz:=false port:=11348" &

sleep 10

xterm -hold -e "export ROS_MASTER_URI=http://localhost:11314; export ROS_HOSTNAME=localhost; roslaunch pclprocess train.launch rviz:=true port:=11349" &

sleep 30

xterm -hold -e "python3 ~/drl_ws/train_parallel.py" &

exec "$@"
