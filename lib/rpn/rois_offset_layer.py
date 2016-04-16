# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from utils.cython_bbox import bbox_overlaps
DEBUG = False

class RoisOffsetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._time_steps = bottom[0].data.shape[0]
        assert(bottom[0].data.shape[1] == bottom[1].data.shape[0])
        # sampled rois (0, x1, y1, x2, y2)
        num_rois = bottom[0].data.shape[1]
        top[0].reshape(self._time_steps * num_rois, 5)

    def forward(self, bottom, top):

        pred_offset = bottom[0].data
        num_rois = bottom[0].data.shape[1]
        rois = bottom[1].data
        gt_boxes = bottom[2].data
        im_info = bottom[3].data[0,:]
        overlaps = bbox_overlaps(
        np.ascontiguousarray(rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)

    

        

        # new rois
        top[0].reshape(self._time_steps * num_rois, 5)
        top[0].data[:,0] = 0
        #copy rois to the first time step of rois_offset
        top[0].data[:num_rois,:] = rois
        #copy the adjust rois with time step 0 --> time_steps-1
        for t in xrange(self._time_steps-1):
            rois_offset = _compute_rois_offset(
                    rois[:, 1:5], gt_boxes[gt_assignment, :4],pred_offset[t,:,:], im_info)
            top[0].data[(t + 1) * num_rois : (t + 2) * num_rois, 1:5] = rois_offset
        

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _compute_rois_offset(ex_rois, gt_rois, offset, im_info):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4
    assert offset.shape[1] == 4
    assert ex_rois.shape[0] == offset.shape[0]
    
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev -- reverse the transformation
        offset = offset * np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS) + np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    rois_offset = bbox_transform_inv(gt_rois, offset)         
    rois_offset = clip_boxes(rois_offset, im_info[:2])
    return rois_offset