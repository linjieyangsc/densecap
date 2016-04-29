 
GPU_ID=1
NET_FINAL=output/faster_rcnn_end2end/vg_train/faster_rcnn_cap_two_stage_reg_iter_300000.caffemodel
TEST_IMDB="vg_test_subset"
PT_DIR="faster_rcnn_cap"
time ./lib/tools/test_net_cap.py --gpu ${GPU_ID} \
  --def_feature models/${PT_DIR}/vgg_deploy_two_stage.prototxt \
  --def_recurrent models/${PT_DIR}/test_cap.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg models/${PT_DIR}/faster_rcnn_end2end.yml \