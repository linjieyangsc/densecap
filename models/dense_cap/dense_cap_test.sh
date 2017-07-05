GPU_ID=3
NET_FINAL=output/dense_cap/vg_1.0_train/dense_cap_joint_inference_iter_200.caffemodel
TEST_IMDB="vg_1.0_val"
PT_DIR="dense_cap"
time ./lib/tools/test_net.py --gpu ${GPU_ID} \
  --def_feature models/${PT_DIR}/vgg_region_global_feature.prototxt \
  --def_recurrent models/${PT_DIR}/test_cap_pred_no_context.prototxt \
  --def_embed models/${PT_DIR}/test_word_embedding.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg models/${PT_DIR}/dense_cap.yml \
