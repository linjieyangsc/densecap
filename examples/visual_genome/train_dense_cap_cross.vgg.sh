#!/usr/bin/env bash

GPU_ID=6
WEIGHTS=\
./models/vggnet/VGG_ILSVRC_16_layers.caffemodel

./build/tools/caffe train \
    -solver ./examples/visual_genome/dense_cap_cross_solver.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
