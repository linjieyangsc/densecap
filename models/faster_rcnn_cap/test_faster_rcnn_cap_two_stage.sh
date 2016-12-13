GPU_ID=1
NET_FINAL=output/faster_rcnn_end2end/vg_1.0_train/faster_rcnn_cap_two_stage_context8_finetune3_iter_200000.caffemodel
TEST_IMDB="vg_1.0_test_subset"
PT_DIR="faster_rcnn_cap"
time ./lib/tools/test_net_cap_two_stage.py --gpu ${GPU_ID} \
  --def_feature models/${PT_DIR}/vgg_region_global2_feature_512.prototxt \
  --def_recurrent models/${PT_DIR}/test_cap_pred_context8.prototxt \
  --def_embed models/${PT_DIR}/test_word_embedding_512.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
	--vis \
  --cfg models/${PT_DIR}/faster_rcnn_end2end.yml \
