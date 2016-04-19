 
GPU_ID=1
NET_FINAL=output/faster_rcnn_end2end/vg_train/faster_rcnn_cap_iter_300000.caffemodel
TEST_IMDB="vg_test"
PT_DIR="faster_rcnn_cap"
time ./lib/tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/train_cap.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg models/${PT_DIR}/faster_rcnn_end2end.yml \