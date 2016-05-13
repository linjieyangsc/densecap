#!/usr/bin/env bash

GPU_ID=1
WEIGHTS=./models/vggnet/VGG_ILSVRC_16_layers.caffemodel


./build/tools/caffe train \
    -solver ./models/lstm/lrcn2_solver.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
