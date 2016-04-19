#!/usr/bin/env bash

GPU_ID=2
#WEIGHTS=./models/lstm/sent_lstm_lm_iter_100000.caffemodel
./build/tools/caffe train \
    -solver ./models/lstm/lstm_lm_new_solver.prototxt \
    -gpu $GPU_ID
