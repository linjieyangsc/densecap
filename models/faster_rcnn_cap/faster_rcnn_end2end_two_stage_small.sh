# Example:
# ./models/faster_rcnn_cap/faster_rcnn_end2end_two_stage.sh 1 visual_genome SOLVER_NAME
# ./models/faster_rcnn_cap/faster_rcnn_end2end_two_stage.sh 1 visual_genome SOLVER_NAME PRETRAINED_MODEL_PATH

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
SOLVER=$3

array=( $@ )
len=${#array[@]}
# finetuning CNN, needs model path 
if [ $len -eq 4 ]; then
WEIGHTS=$4
EXTRA_ARGS=${array[@]:4:$len}
else
# finetuning from vgg model, fixing CNN
WEIGHTS=models/vggnet/VGG_ILSVRC_16_layers.caffemodel
EXTRA_ARGS=${array[@]:3:$len}
fi
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
case $DATASET in
  visual_genome)
    TRAIN_IMDB="vg_1.0_train"
    TEST_IMDB="vg_1.0_val"
    PT_DIR="faster_rcnn_cap"
    ITERS=200000
    ;;
  visual_genome_1.2)
    TRAIN_IMDB="vg_1.2_train"
    TEST_IMDB="vg_1.2_val"
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
  --weights ${WEIGHTS} \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg models/${PT_DIR}/faster_rcnn_end2end_small.yml \
  ${EXTRA_ARGS}



