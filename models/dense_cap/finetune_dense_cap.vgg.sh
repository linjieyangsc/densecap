#!/usr/bin/env bash

GPU_ID=4
WEIGHTS=\
./models/dense_cap/dense_cap_iter_300000.caffemodel

./build/tools/caffe train \
    -solver ./models/dense_cap/dense_cap_finetune_solver.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
