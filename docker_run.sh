#! /bin/bash
:  '
usuage instructions :
to run this script you should first grant it execution using:
    chmod +x docker_run.sh

then you can give it a name as a first argument and a tensorflow 
version as a second argument :
    ./run_docker.sh docker_container_name 1
    
Choose 1 for previous tensorflow image and 2 for tensorflow new 
generation
'


docker_name=$1
tensorflow_version=$2

if [ $tensorflow_version == 1 ]; then
    image="tensorflow/tensorflow:1.13.1-gpu-py3"
elif [ $tensorflow_version == 3 ]; then
    image="ai_platform:latest"
else
    image="tensorflow/tensorflow:latest-gpu-py3"
fi

echo "Docker is running TensorFlow version $tensorflow_version as $docker_name"

nvidia-docker run -it \
    --rm \
    -u $(id -u):$(id -g) \
    --env="DISPLAY" \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --workdir="/home/$USER" \
    --volume="/home/$USER:/home/$USER" \
    --device /dev/video0 \
    --network host \
    --name $docker_name \
    $image \
    bash