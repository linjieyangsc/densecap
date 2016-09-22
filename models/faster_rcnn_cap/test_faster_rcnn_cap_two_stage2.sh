GPU_ID=0
NET_FINAL=output/faster_rcnn_end2end/vg_1.0_train/faster_rcnn_cap_two_stage_context6_finetune3_iter_200000.caffemodel
TEST_IMDB="vg_1.0_test"
PT_DIR="faster_rcnn_cap"
time ./lib/tools/test_net_cap_two_stage.py --gpu ${GPU_ID} \
  --def_feature models/${PT_DIR}/vgg_region_global2_feature_512.prototxt \
  --def_recurrent models/${PT_DIR}/test_cap_pred_two_lstm_context_concat.prototxt \
  --def_embed models/${PT_DIR}/test_word_embedding_512.prototxt \
  --net ${NET_FINAL} \
  --use_box_at 2 \
  --imdb ${TEST_IMDB} \
  --cfg models/${PT_DIR}/faster_rcnn_end2end.yml \
