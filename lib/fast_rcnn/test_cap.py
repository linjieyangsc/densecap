# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import math
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import json
from utils.blob import im_list_to_blob
import os
import sys
sys.path.append('models/dense_cap/')
from run_experiment_vgg_vg import gt_region_merge, get_bbox_coord
from vg_to_hdf5_data import *
#sys.path.add('examples/coco-caption')
#import
COCO_EVAL_PATH = '/media/researchshare/linjie/data/MS_COCO/coco-caption/'
sys.path.append(COCO_EVAL_PATH)
from pycocoevalcap.vg_eval import VgEvalCap
eps = 1e-10
DEBUG=False
def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors
def _greedy_search(net, forward_args, proposal_n, max_timestep = 15):
    """Do greedy search to find the regions and captions"""
    # Data preparation
    

    pred_captions = [None] * proposal_n
    pred_locations = [None] * proposal_n
    pred_logprobs = [0.0] * proposal_n
    
    # first time step
    
    #forward_args['']
    forward_args['cont_sentence'] = np.zeros((1,proposal_n))
    forward_args['input_sentence'] = np.zeros((1,proposal_n)) # start with EOS
    # reshape blobs
    for k, v in forward_args.iteritems():
        if DEBUG:
            print 'shape of %s is ' % k
            print v.shape
        net.blobs[k].reshape(*(v.shape))
 

    for step in xrange(max_timestep):
        blobs_out = net.forward(**forward_args)

        pred_location = blobs_out['predict_loc'].reshape(proposal_n, 4)# 1 x proposal_n x 4 --> proposal_n x 4
        word_probs = blobs_out['probs'].copy()
            
        #suppress <unk> tag
        word_probs[:,:,1] = 0
        best_words = word_probs.argmax(axis = 2).reshape(proposal_n)
        if DEBUG:
            if step == 0:
                print 'content in blobs_out'
                for k,v in blobs_out.iteritems():
                    print k
                print 'shape of word probs' 
                print word_probs.shape
                print 'image features'
                print forward_args['image_features']
                print 'mean value'
                print forward_args['image_features'].mean()
                print 'roi pooling features'
                print net.blobs['pool5'].data[:2,:,:]
                print 'mean value'
                print net.blobs['pool5'].data.mean()                
                print 'sample data of pooled features'
                print net.blobs['fc6'].data[:2,:]
                print 'mean value'
                print net.blobs['fc6'].data.mean()
            print 'step %d' % step
            print 'sample data of output of lstm1'
            print net.blobs['lstm1'].data[0,:,:10]
            print 'word probs for the first 10'
            print word_probs[0,:,:10]
        for i, w, loc in zip(range(proposal_n), best_words, pred_location):
            if not pred_captions[i]:
                pred_captions[i] = [w]
                pred_locations[i] = [loc]
                pred_logprobs[i] = math.log(word_probs[0,i,w] + eps)
            elif pred_captions[i][-1] != 0:
                pred_captions[i].append(w)
                pred_locations[i].append(loc)
                pred_logprobs[i] += math.log(word_probs[0,i,w] + eps)
        forward_args['input_sentence'][:] = best_words
        forward_args['cont_sentence'][:] = 1
        
    #transform location sequence to numpy matrix
    for i in xrange(proposal_n):
        pred_locations[i] = np.array(pred_locations[i])
    return pred_captions, pred_locations, pred_logprobs

def im_detect(feature_net, recurrent_net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        feature_net (caffe.Net): CNN model for extracting features
        recurrent_net (caffe.Net): Recurrent model for generating captions and locations
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x 1 array of object class scores 
        boxes_seq (list): length R list of caption_n x 4 array of predicted bounding boxes
        captions (list): length R list of length caption_n list of word tokens (captions)
    """
    # Previously:
    # 1. forward pass of one image
    # 2. get rois, bbox score and bbox prediction
    # Now:
    # 1. forward pass of one image --> image features and a list of proposals (rois)
    # 2. for each proposal, do greedy search, which is the same way as DenseCap
    # 
    blobs, im_scales = _get_blobs(im, boxes)
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    # reshape network inputs
    feature_net.blobs['data'].reshape(*(blobs['data'].shape))
    feature_net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    
    #proposal_n = something here
    feature_net.forward(data = im_blob, im_info = blobs['im_info'])
    image_features = feature_net.blobs['conv5_3'].data.copy()
    rois = feature_net.blobs['rois'].data.copy()
    boxes = rois[:, 1:5] / im_scales[0]
    proposal_n = rois.shape[0]
    feat_args = {'image_features': image_features, 'rois': rois}
    # do greedy search
    
    captions, locations, logprobs = _greedy_search(recurrent_net, feat_args, proposal_n)

    
    assert len(im_scales) == 1, "Only single-image batch implemented"
    

    
    # use rpn scores, combine with caption score later
    scores = feature_net.blobs['rpn_scores'].data.copy()

    #bbox transform
    #stacking

    boxes_stack = np.concatenate([np.tile(box,(loc.shape[0],1)) for box, loc in zip(boxes, locations)])
    box_deltas_stack = np.concatenate(locations) # horizontally stacking
    group_ids = np.array([loc.shape[0] for loc in locations]).cumsum()
    group_ids = np.insert(group_ids, 0, 0)
    #do the transformation
    pred_boxes_stack = bbox_transform_inv(boxes_stack, box_deltas_stack)
    pred_boxes_stack = clip_boxes(pred_boxes_stack, im.shape)
    #unraveling
    pred_boxes_seq = [None] * proposal_n
    
    for i in xrange(proposal_n):
        pred_boxes_seq[i] = pred_boxes_stack[group_ids[i]:group_ids[i+1],:]
   
    return scores, pred_boxes_seq, captions

def vis_detections(im_path, im, captions, dets, thresh=0.3, save_path='vis'):
    """Visual debugging of detections by saving images with detected bboxes."""
    if not os.path.exists(save_path):
                os.makedirs(save_path)
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        caption = captions[i]
        if score > thresh:
            im_new = np.copy(im)
            im_name = im_path.split('/')[-1][:-4]
            cv2.rectangle(im_new, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,0,0), 2)
            cv2.imwrite('%s/%s_%s_%.2f.jpg' % (save_path, im_name, caption, score), im_new)
            

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def sentence(vocab, vocab_indices):
    # consider <eos> tag with id 0 in vocabulary
    sentence = ' '.join([vocab[i] for i in vocab_indices])
    suffix = ' ' + vocab[0]
    if sentence.endswith(suffix):
      sentence = sentence[:-len(suffix)]
    return sentence

def test_net(feature_net, recurrent_net, imdb, max_per_image=100, thresh=0.01, vis=True):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    if DEBUG:
        print 'number of images: %d' % num_images
    # all detections are collected into:
    #    all_regions[image] = list of {'image_id', caption', 'location', 'location_seq'}
    all_regions = [None] * num_images

    output_dir = get_output_dir(imdb, feature_net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb
    #read vocabulary  & add <eos> tag
    vocab = list(imdb.get_vocabulary())
    vocab.insert(0, '<EOS>')

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes_seq, captions = im_detect(feature_net, recurrent_net, im, box_proposals)
        #features = extract_feature(net, im)

        _t['im_detect'].toc()

        _t['misc'].tic()
        # only one positive class
        if DEBUG:
            print 'shape of scores'
            print scores.shape
        inds = np.where(scores > thresh)[0]
        pos_scores = scores[inds]
        # get the last predicted box
        pos_boxes_seq = [boxes_seq[idx] for idx in inds]
        pos_boxes = np.array([bs[-1,:] for bs in pos_boxes_seq])
        pos_captions = [captions[idx] for idx in inds]
  
        pos_dets = np.hstack((pos_boxes, pos_scores)) \
            .astype(np.float32, copy=False)
        keep = nms(pos_dets, cfg.TEST.NMS)
        pos_dets = pos_dets[keep, :]
        pos_scores = pos_scores[keep, 0]
        pos_captions = [sentence(vocab, pos_captions[idx]) for idx in keep]
        pos_boxes_seq = [pos_boxes_seq[idx] for idx in keep]
        if vis:
            #TODO(Linjie): display location sequence
            vis_detections(imdb.image_path_at(i), im, pos_captions, pos_dets, save_path = os.path.join(output_dir,'vis'))
        all_regions[i] = []
        #follow the format of baseline models routine
        for cap, box_seq, prob in zip(pos_captions, pos_boxes_seq, pos_scores):
            #region sizes normalize to [0, 1], follow baseline model routine
            #box_seq[:, [0, 2]] /= im.shape[1]
            #box_seq[:, [1, 3]] /= im.shape[0]
            anno = {'image_id':i, 'prob': format(prob,'.3f'), 'caption':cap, \
            'location_seq': box_seq.tolist(), 'location': box_seq[-1,:].tolist()}
            all_regions[i].append(anno)

        
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)
    generation_result = [{
      'image_id': image_index,
      'image_path': imdb.image_path_at(image_index),
      'caption_locations': all_regions[image_index]
    } for image_index in xrange(num_images)]
  
    det_file = os.path.join(output_dir, 'generation_result.json')
    with open(det_file, 'w') as f:
        json.dump(generation_result, f)

    print 'Evaluating detections'
    #imdb.evaluate_detections(all_regions, output_dir)
   
   
    
    #gt_regions = imdb.get_gt_regions() # is a list
    gt_regions_merged = [None] * num_images
    #transform gt_regions into the baseline model routine
    #for i,regions in enumerate(gt_regions):
    for i, image_index in enumerate(imdb.image_index):
        new_gt_regions = []
        regions = imdb.get_gt_regions_index(image_index)
        for reg in regions['regions']:
            #if DEBUG:
                #print 'region info'
                #print reg
            loc = np.array([reg['x'], reg['y'], reg['x'] + reg['width'], reg['y'] + reg['height']])
            anno = {'image_id':i, 'caption': reg['phrase'], 'location': loc}
            new_gt_regions.append(anno)
        #merge regions with large overlapped areas
        assert(len(new_gt_regions) > 0)
        gt_regions_merged[i] = gt_region_merge(new_gt_regions)
    image_ids = range(num_images)
    vg_evaluator = VgEvalCap(gt_regions_merged, all_regions)
    vg_evaluator.params['image_id'] = image_ids
    vg_evaluator.evaluate()
    