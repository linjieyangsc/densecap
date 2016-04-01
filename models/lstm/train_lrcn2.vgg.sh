#!/usr/bin/env bash

GPU_ID=10
WEIGHTS=./models/lstm/lrcn2_vgg_cont_iter_100000.caffemodel
#./models/vggnet/VGG_ILSVRC_16_layers.caffemodel


./build2/tools/caffe train \
    -solver ./models/lstm/lrcn2_solver.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
