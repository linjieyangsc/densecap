#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test_cap_two_stage import test_net
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
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def_feature', dest='feature_prototxt',
                        help='prototxt file defining the network (for extracting feature)',
                        default=None, type=str)
    parser.add_argument('--def_recurrent', dest='recurrent_prototxt',
                        help='prototxt file defining the network (for captioning generation)',
                        default=None, type=str)
    parser.add_argument('--def_embed', dest='embed_prototxt',
                        help='prototxt file defining the network (for word embedding)',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='vg_1.0_test', type=str)
   
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--use_box_at', dest='use_box_at',
                        help='use predicted box at this time step, fault to the last',
                        default=-1, type=int)
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

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    feature_net = caffe.Net(args.feature_prototxt, args.caffemodel, caffe.TEST)
    embed_net = caffe.Net(args.embed_prototxt, args.caffemodel, caffe.TEST)
    recurrent_net = caffe.Net(args.recurrent_prototxt, args.caffemodel, caffe.TEST)
    feature_net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    imdb = get_imdb(args.imdb_name)
    #print args.max_per_image
    test_net(feature_net, embed_net, recurrent_net, imdb, \
        vis=args.vis, use_box_at=args.use_box_at)
