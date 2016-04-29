# --------------------------------------------------------
# Linjie Yang
# 04/21/2016
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from rpn.rois_offset_layer import compute_rois_offset
DEBUG = False

class BBoxTargetsOffsetLayer(caffe.Layer):
    """
    Using the predicted bbox offsets to recalculate the offset from ground truth bboxes to the region of interests. 
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        num_rois_offset = bottom[0].data.shape[0]
        num_rois = bottom[1].data.shape[0]
        assert(num_rois_offset % num_rois == 0) # times of num_rois
        self._time_steps = num_rois_offset / num_rois
        
        # shifted bbox targets
        top[0].reshape(self._time_steps, num_rois, 4)

    def forward(self, bottom, top):

        rois_offset = bottom[0].data
        bbox_targets = bottom[1].data
        num_rois = bbox_targets.shape[0]
        #num_rois_offset = rois_offset.shape[0]
        
        #time_steps = num_rois_offset / num_rois
        # get original target bboxes
        rois_original = rois_offset[:num_rois, 1:5]
        target_bboxes = compute_rois_offset(rois_original, bbox_targets)
        # new targets
        top[0].reshape(self._time_steps, num_rois, 4)
        
        #copy bbox targets to the first time step of bbox_targets_offset
        top[0].data[0,:,:] = bbox_targets
        #copy the adjust bbox targets with time step 1 --> time_steps
        for t in xrange(1, self._time_steps):
            targets_prev = top[0].data[t,:,:]
            rois_step = rois_offset[t * num_rois: (t+1) * num_rois,1:5]
            targets_offset = _compute_targets(rois_step, target_bboxes)
            if DEBUG:
                shifted_bboxes = compute_rois_offset(rois_step, targets_offset)
                print 'check bbox consistency'
                print shifted_bboxes[:2,:]
                print target_bboxes[:2,:]
                #print np.linalg.norm(shifted_bboxes - target_bboxes)
                assert np.linalg.norm(shifted_bboxes - target_bboxes) < 0.01
            top[0].data[t,:,:] = targets_offset
        

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return targets