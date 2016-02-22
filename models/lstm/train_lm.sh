#!/usr/bin/env bash

GPU_ID=3
WEIGHTS=./models/lstm/sent_lstm_lm_iter_100000.caffemodel
./build/tools/caffe train \
    -solver ./models/lstm/lstm_lm_solver.prototxt \
    -weights $WEIGHTS\
    -gpu $GPU_ID
