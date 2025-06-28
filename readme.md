# DRL Robot Navigation with TD3
This repository is adapted from **ReiniCimurs** DRL-robot-navigation to train my own Robot Model.

if you're using docker

sudo docker run -it --rm --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw  -v "$(pwd)/sb3_logs_rovid_final:/root/drl_ws/sb3_logs_rovid_final" -v $HOME/.Xauthority:/root/.Xauthority:rw --name rovid_debug rovid_trainer roslaunch pclprocess 


python3 -m ensurepip --upgrade
pip3 install tensorboard

python3 -m tensorboard.main --logdir sb3_logs_rovid_final

if you are using vast.ai, you must setup dns. otherwise, docker cannot download your ros file because it was blocked

docker build -t ros-noetic-rl .

---

## Sensors Used
* RPLidar A2M8
* Intel Realsense L515

---

## Installation

**Requirements:**
* Ubuntu 20.04
* ROS Noetic
* Pytorch
* Tensorboard

---

## Training in your Ubuntu
**Step**
1. Clone this Repository 
    ```bash
    git clone https://github.com/rafaalfe/DRL-robot-navigation
    ```
2. Go to catkin workspace
    ```bash
    cd /DRL-robot-navigation/catkin_ws
    ```
3. Build isolated the catkin workspace
    ```bash
    catkin_make_isolated
    ```
4. Prepare environment training with this command, if you are using multiple simulation you have to assign unique URI. **For example** if you want to do 2 simulation, at first terminal assign ROS_MASTER_URI=http://localhost:11311, port:=11346 and at the second terminal assign ROS_MASTER_URI=http://localhost:11312, port:=11347. You have to assign **cpu_num** at *train_parallel.py* 
    ```bash
    export ROS_MASTER_URI=http://localhost:11311
    export ROS_HOSTNAME=localhost
    export GAZEBO_RESOURCE_PATH=~/Downloads/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
    source ~/Downloads/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash 
    roslaunch pclprocess train.launch rviz:=false port:=11346
    ```
5. Start your Training in the new teminal 
    ```bash
    python3 train_parallel.py
    ```
---

## Training with Docker
**Step**
1. Clone this Repository 
    ```bash
    git clone https://github.com/rafaalfe/DRL-robot-navigation
    ```
2. 