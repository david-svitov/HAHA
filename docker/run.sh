#!/usr/bin/env bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${CURRENT_DIR}/source.sh

NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') docker run -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all -ti $PARAMS $VOLUMES $NAME $@
