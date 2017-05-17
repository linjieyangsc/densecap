GPU_ID=2
NET_FINAL=output/faster_rcnn_end2end/vg_1.0_train/faster_rcnn_cap_joint_one_lstm_context_ef_concat_finetune_iter_200000.caffemodel
TEST_IMDB="vg_1.0_test"
PT_DIR="faster_rcnn_cap"
time ./lib/tools/test_net_cap_two_stage_fusion.py --gpu ${GPU_ID} \
  --def_feature models/${PT_DIR}/vgg_region_global2_feature_512.prototxt \
  --def_recurrent models/${PT_DIR}/test_cap_pred_joint_one_lstm_context_ef.prototxt \
  --def_embed models/${PT_DIR}/test_word_embedding_512.prototxt \
	--def_fusion models/${PT_DIR}/test_feature_fusion.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg models/${PT_DIR}/faster_rcnn_end2end.yml \
