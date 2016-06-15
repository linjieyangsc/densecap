# Do freeze-convnet training first, then finetuning
# Example:
# ./models/faster_rcnn_cap/faster_rcnn_end2end_train_finetune.sh 1 visual_genome MODEL_TYPE

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
MODEL_TYPE=$3

array=( $@ )
len=${#array[@]}
WEIGHTS=models/vggnet/VGG_ILSVRC_16_layers.caffemodel
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
case $DATASET in
   visual_genome)
    TRAIN_IMDB="vg_1.0_train"
    TEST_IMDB="vg_1.0_val"
    PT_DIR="faster_rcnn_cap"
    FINETUNE_AFTER=200000
    ITERS=400000
    ;;
  visual_genome_1.2)
    TRAIN_IMDB="vg_1.2_train"
    TEST_IMDB="vg_1.2_val"
    PT_DIR="faster_rcnn_cap"
    FINETUNE_AFTER=200000
    ITERS=400000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac
# Training
GLOG_logtostderr=1
./lib/tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/solver_${MODEL_TYPE}.prototxt \
  --weights ${WEIGHTS} \
  --imdb ${TRAIN_IMDB} \
  --iters ${FINETUNE_AFTER} \
  --cfg models/${PT_DIR}/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
NEW_WEIGHTS=output/faster_rcnn_end2end/vg_train/faster_rcnn_cap_${MODEL_TYPE}_iter_${ITERS}.caffemodel
# Finetuning
./lib/tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/solver_${MODEL_TYPE}_finetune.prototxt \
  --weights ${NEW_WEIGHTS} \
  --imdb ${TRAIN_IMDB} \
  --iters `expr ${ITERS} - ${FINETUNE_AFTER}` \
  --cfg models/${PT_DIR}/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

