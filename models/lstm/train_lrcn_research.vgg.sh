#!/usr/bin/env bash

GPU_ID=1
WEIGHTS=\
./models/vggnet/VGG_ILSVRC_16_layers.caffemodel
DATA_DIR=./models/lstm/h5_data/
if [ ! -d $DATA_DIR ]; then
    echo "Data directory not found: $DATA_DIR"
    echo "First, download the COCO dataset (follow instructions in data/coco)"
    echo "Then, run ./examples/coco_caption/coco_to_hdf5_data.py to create the Caffe input data"
    exit 1
fi

../caffe-research/build/tools/caffe train \
    -solver ./models/lstm/lrcn_solver.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
