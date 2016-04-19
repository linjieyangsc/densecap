#!/usr/bin/env bash

GPU_ID=7
#WEIGHTS=./models/lstm/sent_lstm_lm_iter_100000.caffemodel
./build/tools/caffe train \
    -solver ./models/lstm/lstm_lm_par_solver.prototxt \
    -gpu $GPU_ID
