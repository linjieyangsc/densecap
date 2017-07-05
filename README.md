## Dense Captioning with Joint Inference and Visual Context ##
This repo is the released code of dense image captioning models described in the CVPR 2017 paper:
```
 @InProceedings{CVPR17,
  author       = "Linjie Yang and Kevin Tang and Jianchao Yang and Li-Jia Li",
  title        = "Dense Captioning with Joint Inference and Visual Context",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month        = "Jul",
  year         = "2017"
}
```
All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. When using the code in your research work, please cite the above paper.
Our code is adapted from the popular [Faster-RCNN repo](https://github.com/rbgirshick/py-faster-rcnn) written by Ross Girshick. The evaluation code is adapted from [COCO captioning evaluation code](https://github.com/tylin/coco-caption).


## Compiling ##

### Compile Caffe ###
Please follow [official guide](http://caffe.berkeleyvision.org/). Support CUDA 7.5+, CUDNN 5.0+. Tested on Ubuntu 14.04.
### Compile local libraries ###
```
cd lib
make
```
## Demo ##
Download official sample model using script `download_model.sh` in `models/dense_cap`.
Test model with an input image:
```
python ./lib/tools/demo.py --image [IMAGE_PATH] --gpu [GPU_ID]
```
It will generate a folder named "demo" in the library root. Inside the "demo" folder, there will be an HTML page showing the predicted results.
## Training ##
### Data preparation ###
For model training you will need to download the visual genome dataset from [Visual Genome Website](http://visualgenome.org/api/v0/api_home.html), either 1.0 or 1.2 is fine.
Download pre-trained VGG16 model from [link](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel).
Modify data paths in `models/dense_cap/preprocess.py` and run it from the library root to generate training/validation/testing data.

### Start training ###
Run `models/dense_cap/dense_cap_train.sh` to start training. For example, to train a model with joint inference and visual context (late fusion, feature summation) on visual genome 1.0:
```
./models/dense_cap/dense_cap_train.sh [GPU_ID] visual_genome late_fusion_sum [VGG_MODEL_PATH] 
```
It typically takes 2 days to finish training.
## Evaluation ##
Modify `models/dense_cap/dense_cap_test.sh` according to the model you want to test. For example, if you want to test the late-fusion context model with summation, it will look like this:
```
GPU_ID=3
NET_FINAL=output/dense_cap/vg_1.0_train/dense_cap_late_fusion_sum_finetune_iter_200000.caffemodel
TEST_IMDB="vg_1.0_test"
PT_DIR="dense_cap"
time ./lib/tools/test_net.py --gpu ${GPU_ID} \
  --def_feature models/${PT_DIR}/vgg_region_global_feature.prototxt \
  --def_recurrent models/${PT_DIR}/test_cap_pred_context.prototxt \
  --def_embed models/${PT_DIR}/test_word_embedding.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg models/${PT_DIR}/dense_cap.yml \
```
Except the model path(`NET_FINAL`), the only thing you should change is `def_recurrent`, which should be `models/${PT_DIR}/test_cap_pred_no_context.prototxt` for models without context information and `models/${PT_DIR}/test_cap_pred_context.prototxt` for models with context information.
To visualize the result, you can add `--vis` to the end of the above script. It will generate html pages for each image visualizing the results under folder `output/dense_cap/${TEST_IMDB}/vis`.

## Contact ##
If you have any questions regarding the repo, please send email to Linjie Yang (linjie.yang@snap.com).
