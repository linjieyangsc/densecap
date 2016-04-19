#!/usr/bin/env bash

GPU_ID=8
WEIGHTS=\
./models/lstm/lrcn2_vgg_cont2_iter_50000.caffemodel

./build/tools/caffe train \
    -solver ./models/lstm/lrcn2_finetune_solver.vgg.prototxt \
 -gpu 8,9,10,11 #    -weights $WEIGHTS \
   # -gpu 8,9,10,11
