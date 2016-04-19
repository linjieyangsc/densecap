#!/usr/bin/env bash

GPU_ID=6
#WEIGHTS=./models/lstm/sent_lstm_lm_iter_100000.caffemodel
./build/tools/caffe train \
    -solver ./models/lstm/skip_lm_solver.prototxt \
    -gpu $GPU_ID
