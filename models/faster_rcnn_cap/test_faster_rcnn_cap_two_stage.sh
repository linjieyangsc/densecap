GPU_ID=0
NET_FINAL=output/faster_rcnn_end2end/vg_train/faster_rcnn_cap_two_stage2_finetune_iter_200000.caffemodel
TEST_IMDB="vg_1.0_test"
PT_DIR="faster_rcnn_cap"
time ./lib/tools/test_net_cap_two_stage.py --gpu ${GPU_ID} \
  --def_feature models/${PT_DIR}/vgg_region_feature.prototxt \
  --def_recurrent models/${PT_DIR}/test_cap_pred2.prototxt \
  --def_embed models/${PT_DIR}/test_word_embedding.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg models/${PT_DIR}/faster_rcnn_end2end.yml \
