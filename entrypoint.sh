#!/bin/bash
set -e

# Setup environment ROS setiap kali kontainer dimulai
source "/opt/ros/noetic/setup.bash"
export GAZEBO_RESOURCE_PATH=/root/catkin_ws/src/multi_robot_scenario/launch
source "/root/catkin_ws/devel/setup.bash"

# Start a new detached tmux session if one isn't already running
tmux has-session -t ros_sessions || tmux new-session -d -s ros_sessions

# Function to start a roslaunch in a new tmux window
start_roslaunch_in_tmux() {
    local window_name=$1
    local ros_master_uri=$2
    local ros_hostname=$3
    local roslaunch_cmd=$4

    tmux new-window -t ros_sessions -n "$window_name" "$ros_master_uri $ros_hostname $roslaunch_cmd"
}

# Roslaunch commands
start_roslaunch_in_tmux "pcl_11346" "export ROS_MASTER_URI=http://localhost:11311;" "export ROS_HOSTNAME=localhost;" "roslaunch pclprocess train.launch rviz:=false port:=11346" &
sleep 10

start_roslaunch_in_tmux "pcl_11347" "export ROS_MASTER_URI=http://localhost:11312;" "export ROS_HOSTNAME=localhost;" "roslaunch pclprocess train.launch rviz:=false port:=11347" &
sleep 10

start_roslaunch_in_tmux "pcl_11348" "export ROS_MASTER_URI=http://localhost:11313;" "export ROS_HOSTNAME=localhost;" "roslaunch pclprocess train.launch rviz:=false port:=11348" &
sleep 10

start_roslaunch_in_tmux "pcl_11349_rviz" "export ROS_MASTER_URI=http://localhost:11314;" "export ROS_HOSTNAME=localhost;" "roslaunch pclprocess train.launch rviz:=true port:=11349" &
sleep 30

# Start the Python training script in another tmux window
tmux new-window -t ros_sessions -n "train_parallel" "python3 ~/drl_ws/train_parallel.py" &

# You can then attach to the tmux session to monitor: `tmux attach -t ros_sessions`
# Or list windows: `tmux list-windows -t ros_sessions`

exec "$@"