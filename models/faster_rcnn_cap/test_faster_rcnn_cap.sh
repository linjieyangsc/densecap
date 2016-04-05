GPU_ID=$1
NET_FINAL=$2
TEST_IMDB="vg_val"
PT_DIR="faster_rcnn_cap"
time ./lib/tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg models/${PT_DIR}/faster_rcnn_end2end.yml \