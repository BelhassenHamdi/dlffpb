#! /bin/bash

docker run -u $(id -u):$(id -g) -v $PWD:/tmp jrottenberg/ffmpeg:4.0-ubuntu -stats -i /tmp/DiningEntrance_2019-01-22_19-00-01.mkv -c:v libx265 -pix_fmt yuv420p10 -f mp4 /tmp/DiningEntrance_2019-01-22_19-00-01.mp4
