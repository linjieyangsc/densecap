#!/usr/bin/env bash

GPU_ID=9
WEIGHTS=\
./models/lstm/lrcn2_finetune_vgg_iter_100000.caffemodel

./build/tools/caffe test \
    -model ./models/lstm/lrcn2_test.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
