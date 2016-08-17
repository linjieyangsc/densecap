"""Test a network on an imdb, results loaded from file."""
import _init_paths
import argparse
import numpy as np
import math
import cv2
import json
import os
import sys
sys.path.append('models/dense_cap/')
from run_experiment_vgg_vg import gt_region_merge, get_bbox_coord
from vg_to_hdf5_data import *
#sys.path.add('examples/coco-caption')
#import
COCO_EVAL_PATH = 'coco-caption/'
sys.path.append(COCO_EVAL_PATH)
from pycocoevalcap.vg_eval import VgEvalCap
from datasets.factory import get_imdb
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--file', dest='json_file',
                        help='json file to be read',
                        default='',type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args       

def test_net(imdb, json_file):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
   
    all_regions = [None] * num_images
    
    det_file = os.path.join(json_file)

    results = json.load(open(det_file))


    for i in xrange(num_images):
        all_regions[i] = []
        #follow the format of baseline models routine
        key = imdb.image_path_at(i).split('/')[-1]
        pos_captions = results[key]['captions']
        pos_scores = np.exp(np.array(results[key]['logprobs']))
        pos_boxes = results[key]['boxes']
        for cap, box, prob in zip(pos_captions, pos_boxes, pos_scores):
            anno = {'image_id':i, 'prob': format(prob,'.3f'), 'caption':cap, \
            'location': box}
            all_regions[i].append(anno)


   
    
    #gt_regions = imdb.get_gt_regions() # is a list
    gt_regions_merged = [None] * num_images
    #transform gt_regions into the baseline model routine
    #for i,regions in enumerate(gt_regions):
    for i, image_index in enumerate(imdb.image_index):
        new_gt_regions = []
        regions = imdb.get_gt_regions_index(image_index)
        for reg in regions['regions']:
            loc = np.array([reg['x'], reg['y'], reg['x'] + reg['width'], reg['y'] + reg['height']])
            anno = {'image_id':i, 'caption': reg['phrase'].encode('ascii','ignore'), 'location': loc}
            new_gt_regions.append(anno)
        #merge regions with large overlapped areas
        assert(len(new_gt_regions) > 0)
        gt_regions_merged[i] = gt_region_merge(new_gt_regions)
    image_ids = range(num_images)
    vg_evaluator = VgEvalCap(gt_regions_merged, all_regions)
    vg_evaluator.params['image_id'] = image_ids
    vg_evaluator.evaluate()


if __name__ == "__main__":
    args = parse_args()
    imdb = get_imdb(args.imdb_name)
    json_file = args.json_file
    test_net(imdb, json_file)
