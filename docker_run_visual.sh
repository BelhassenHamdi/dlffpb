#! /bin/bash

nvidia-docker run -it \
    -u $(id -u):$(id -g) \
    --env="DISPLAY" \
    --workdir="/home/$USER" \
    --volume="/home/$USER:/home/$USER" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --device /dev/video0 \
    --network host \
    --name opencv \
    jjanzic/docker-python3-opencv \
    bash