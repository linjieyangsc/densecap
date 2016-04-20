#!/usr/bin/env bash

GPU_ID=4
WEIGHTS=\
./models/vggnet/VGG_ILSVRC_16_layers.caffemodel

./build/tools/caffe train \
    -solver ./models/dense_cap/dense_cap_solver.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
