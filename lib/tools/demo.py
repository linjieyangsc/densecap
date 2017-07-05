#!/usr/bin/env python
# -------------------------------------------------------- 
# test a dense captioning model
# Code adapted from faster R-CNN project
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


import _init_paths
from fast_rcnn.test import test_im
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a dense captioning network')
    parser.add_argument('--image', dest='im_path', help='input image path',
    			default='', type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def_feature', dest='feature_prototxt',
                        help='prototxt file defining the network (for extracting feature)',
                        default='models/dense_cap/vgg_region_global_feature.prototxt', 
			type=str)
    parser.add_argument('--def_recurrent', dest='recurrent_prototxt',
                        help='prototxt file defining the network (for captioning generation)',
                        default='models/dense_cap/test_cap_pred_context.prototxt', type=str)
    parser.add_argument('--def_embed', dest='embed_prototxt',
                        help='prototxt file defining the network (for word embedding)',
                        default='models/dense_cap/test_word_embedding.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='models/dense_cap/dense_cap_late_fusion_sum.caffemodel', 
			type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', 
			default='models/dense_cap/dense_cap.yml', type=str)
    parser.add_argument('--vocab', dest='vocabulary',
    			help='vocabulary file',
			default='data/visual_genome/1.0/vocabulary.txt', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not os.path.exists(args.caffemodel):
        print('Model file {} not exist'.format(args.caffemodel))
        exit()
    if not os.path.exists(args.vocabulary):
        print('Vocabulary file not exist')
	exit()
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    feature_net = caffe.Net(args.feature_prototxt, caffe.TEST, weights=args.caffemodel)
    embed_net = caffe.Net(args.embed_prototxt, caffe.TEST, weights=args.caffemodel)
    recurrent_net = caffe.Net(args.recurrent_prototxt, caffe.TEST, weights=args.caffemodel)
    vocab=['<EOS>']
    with open(args.vocabulary, 'r') as f:
        for line in f:
            vocab.append(line.strip())
    test_im(feature_net, embed_net, recurrent_net, args.im_path, vocab)
