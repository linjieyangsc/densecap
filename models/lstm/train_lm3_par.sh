#!/usr/bin/env bash

GPU_ID=4
./build/tools/caffe train \
    -solver ./models/lstm/lstm_lm3_par_solver.prototxt \
    -gpu $GPU_ID
