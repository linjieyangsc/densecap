#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./models/faster_rcnn_cap/faster_rcnn_end2end.sh 1 visual_genome --set TRAIN.SCALES "[720]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
SOLVER=$3
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=70000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  visual_genome)
    TRAIN_IMDB="vg_train"
    TEST_IMDB="vg_val"
    PT_DIR="faster_rcnn_cap"
    ITERS=200000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

GLOG_logtostderr=1
./lib/tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${SOLVER} \
  --weights models/vggnet/VGG_ILSVRC_16_layers.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg models/${PT_DIR}/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}



