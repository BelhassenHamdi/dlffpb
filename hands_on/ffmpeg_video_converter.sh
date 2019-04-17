#! /bin/bash

docker run -u $(id -u):$(id -g) -v $PWD:/tmp jrottenberg/ffmpeg:4.0-ubuntu -stats -i /tmp/DiningEntrance_2019-01-22_19-00-01.mkv -c:v libx265 -pix_fmt yuv420p10 -f mp4 /tmp/DiningEntrance_2019-01-22_19-00-01.mp4


# ffmpeg with gpu

docker run --rm -it --runtime=nvidia \
    --volume $PWD:/workspace \
    willprice/nvidia-ffmpeg -i input.mp4 output.avi

# ffmpeg from video to images

nvidia-docker run --rm -it \
    --volume $PWD:/workspace \
    willprice/nvidia-ffmpeg \
      -hwaccel_device 0 \
      -hwaccel cuvid \
      -c:v h264_cuvid \
      -i DiningEntrance_2019-04-09_19-00-02.mkv \
      -c:v hevc_nvenc \
      -vf fps=1
      DE%05d.jpg