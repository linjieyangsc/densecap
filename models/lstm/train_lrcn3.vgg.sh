#!/usr/bin/env bash

GPU_ID=6
WEIGHTS=\
./models/vggnet/VGG_ILSVRC_16_layers.caffemodel
./build/tools/caffe train \
    -solver ./models/lstm/lrcn3_solver.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
