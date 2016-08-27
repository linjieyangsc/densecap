"""Test a network on an imdb, results loaded from file."""

import matplotlib
matplotlib.use('Agg')
import _init_paths
import argparse
import numpy as np
import math
import cv2
import json
import os
import sys
#sys.path.append('models/dense_cap/')
from utils.bbox_utils import region_merge, get_bbox_iou_matrix
from datasets.factory import get_imdb
import matplotlib.pyplot as plt
from collections import Counter
def get_stats(imdb):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
   
    #all_regions = [None] * num_images
    

    all_gt_region_len = []
    #gt_regions_merged = [None] * num_images
    #transform gt_regions into the baseline model routine
    #for i,regions in enumerate(gt_regions):
    max_iou_list = []
    for i, image_index in enumerate(imdb.image_index):
        new_gt_regions = []
        regions = imdb.get_gt_regions_index(image_index)
        for reg in regions['regions']:
            loc = np.array([reg['x'], reg['y'], reg['x'] + reg['width'], reg['y'] + reg['height']])
            all_gt_region_len.append(len(reg['phrase'].split()))
            anno = {'image_id':i, 'caption': reg['phrase'].encode('ascii','ignore'), 'location': loc}
            new_gt_regions.append(anno)
        #merge regions with large overlapped areas
        #assert(len(new_gt_regions) > 0)
        if len(new_gt_regions) > 0:
          gt_regions_merged = region_merge(new_gt_regions)
          gt_bboxes = np.array([x['location'] for x in gt_regions_merged], dtype=np.float32)
          iou_matrix = get_bbox_iou_matrix(gt_bboxes)
          #set self overlap ratio to 0
          for x in xrange(gt_bboxes.shape[0]):
              iou_matrix[x,x] = 0
          bbox_max_iou = iou_matrix.max(axis=0)
          max_iou_list.append(bbox_max_iou)
    max_iou_all = np.concatenate(max_iou_list)
    
    bbox_n = len(max_iou_all)
    max_iou_filt = max_iou_all[max_iou_all<=0.7]
    #max_iou_all.sort() 
    #counts, base = np.histogram(max_iou_all, bins=7, range=(0,0.7))
    #counts = counts / float(bbox_n)
    #base = base[::-1]
    #cum_counts = np.cumsum(counts[::-1])
    #plt.plot(base[:-1], counts, c='blue')
    
    plt.hist(max_iou_filt, 7, normed=0, facecolor='#7f7f7f')
    #plt.title('Max IoU between ground truth bounding boxes')
    #plt.bar(base, counts, 0.7, color='b')
    plt.grid(True)
    plt.xlabel('max IoU')
    plt.ylabel('bbox number')
    plt.xlim([0,0.7])
    plt.show()
    save_name = 'bbox_overlap_stats.png'
    plt.savefig(save_name)

    #gt phrase len stats
    fig = plt.figure()
    ax = fig.add_subplot(111)
    all_gt_region_len = np.array(all_gt_region_len)
    region_len_stats = Counter(all_gt_region_len)
    for k,v in dict(region_len_stats).iteritems():
      print k,v
    ax.hist(all_gt_region_len,10)
    fig.show()
    save_name_phrase_len = 'gt_phrase_len.png'
    fig.savefig(save_name_phrase_len)

if __name__ == "__main__":
    
    imdb = get_imdb('vg_1.0_train')
    
    get_stats(imdb)
