#!/usr/bin/env bash

GPU_ID=5
WEIGHTS=\
./examples/visual_genome/dense_cap2_iter_300000.caffemodel

./build/tools/caffe train \
    -solver ./examples/visual_genome/dense_cap2_finetune_solver.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
