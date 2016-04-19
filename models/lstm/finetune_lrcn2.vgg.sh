#!/usr/bin/env bash

GPU_ID=9
WEIGHTS=\
./models/lstm/lrcn2_vgg_cont2_iter_50000.caffemodel

./build/tools/caffe train \
    -solver ./models/lstm/lrcn2_finetune_solver.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
