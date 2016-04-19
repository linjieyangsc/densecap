#!/usr/bin/env bash

GPU_ID=6
./build/tools/caffe train \
    -solver ./models/lstm/lstm_lm2_par_solver.prototxt \
    -gpu $GPU_ID
