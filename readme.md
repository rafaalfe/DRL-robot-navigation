sudo docker run -it --rm --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw  -v "$(pwd)/sb3_logs_rovid_final:/root/drl_ws/sb3_logs_rovid_final" -v $HOME/.Xauthority:/root/.Xauthority:rw --name rovid_debug rovid_trainer roslaunch pclprocess 


python3 -m ensurepip --upgrade
pip3 install tensorboard

python3 -m tensorboard.main --logdir sb3_logs_rovid_final

if you are using vast.ai, you must setup dns. otherwise, docker cannot download your ros file because it was blocked

docker build -t ros-noetic-rl .