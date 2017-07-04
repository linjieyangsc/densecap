GPU_ID=1
NET_FINAL=output/dense_cap/vg_1.0_train/dense_cap_late_fusion_sum_finetune_iter_200000.caffemodel
TEST_IMDB="vg_1.0_test"
PT_DIR="dense_cap"
time ./lib/tools/test_net_cap_two_stage.py --gpu ${GPU_ID} \
  --def_feature models/${PT_DIR}/vgg_region_global_feature.prototxt \
  --def_recurrent models/${PT_DIR}/test_cap_pred_context.prototxt \
  --def_embed models/${PT_DIR}/test_word_embedding.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg models/${PT_DIR}/dense_cap.yml \
