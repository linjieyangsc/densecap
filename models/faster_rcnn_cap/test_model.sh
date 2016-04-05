GPU_ID=3
WEIGHTS=\
./output/faster_rcnn_end2end/vg_train/faster_rcnn_cap_iter_100000.caffemodel
./build/tools/caffe train \
-solver ./models/faster_rcnn_cap/solver.prototxt \
-weights $WEIGHTS \
-gpu $GPU_ID
