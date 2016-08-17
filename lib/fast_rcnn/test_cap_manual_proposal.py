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
COCO_EVAL_PATH = 'coco-caption/'
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

def _greedy_search(embed_net, recurrent_net, forward_args, optional_args, max_timestep = 15, pred_bbox = True):
    """Do greedy search to find the regions and captions"""
    # Data preparation


    pred_captions = [None]
    pred_logprobs = [0.0]
    pred_bbox_offsets = np.zeros((1, 4))
    pred_bbox_offsets_all = []

    
    forward_args['cont_sentence'] = np.zeros((1,1))
    if 'global_features' in optional_args:
        forward_args['global_features'] = optional_args['global_features'].reshape(*(forward_args['input_features'].shape))

    # reshape blobs
    for k, v in forward_args.iteritems():
        if DEBUG:
            print 'shape of %s is ' % k
            print v.shape
        recurrent_net.blobs[k].reshape(*(v.shape))

    recurrent_net.forward(**forward_args)
    forward_args['cont_sentence'][:] = 1

    input_sentence = np.zeros((1,1)) # start with EOS    
    embed_net.blobs['input_sentence'].reshape(1, 1) 
    
    for step in xrange(max_timestep):
        
        embed_out = embed_net.forward(input_sentence=input_sentence)
        forward_args['input_features'] = embed_out['embedded_sentence']
        # another lstm for global features
        if 'global_features' in recurrent_net.blobs:
            forward_args['global_features'] =  embed_out['embedded_sentence']
        blobs_out = recurrent_net.forward(**forward_args)

        word_probs = blobs_out['probs'].copy()
        bbox_pred = blobs_out['bbox_pred'] if pred_bbox else None
        #suppress <unk> tag
        #word_probs[:,:,1] = 0
        best_word = word_probs.argmax(axis = 2).reshape(1)[0]
        finish_n = 0
       
        if not pred_captions[0]:
            pred_captions[0] = [best_word]
            pred_logprobs[0] = math.log(word_probs[0,0,best_word])

        elif pred_captions[0][-1] != 0:
            pred_captions[0].append(best_word)
            pred_logprobs[0] += math.log(word_probs[0,0,best_word])
            pred_bbox_offsets[0,:] = bbox_pred[0,0,:] if pred_bbox else 0
        else:
            break
        if pred_bbox:
            pred_bbox_offsets_all.append(bbox_pred[0,0,:].copy())
        input_sentence[:] = best_word
        forward_args['cont_sentence'][:] = 1
      
    pred_bbox_offsets_all = np.array(pred_bbox_offsets_all)
    return pred_captions, pred_bbox_offsets, pred_logprobs, pred_bbox_offsets_all

def _multi_sampling(embed_net, recurrent_net, forward_args, optional_args, sample_n, temperature=1.0, \
    max_timestep = 15, pred_bbox = True, dedup=True):
    """Do sampling to find the regions and captions"""
    # Data preparation


    pred_captions = [None] * sample_n
    pred_logprobs = [0.0] * sample_n
    pred_bbox_offsets = np.zeros((sample_n, 4))
    eps = 1e-10

    
    forward_args['cont_sentence'] = np.zeros((1,sample_n))
        
    if 'global_features' in optional_args:
        forward_args['global_features'] = optional_args['global_features'].reshape(*(forward_args['input_features'].shape))

    # reshape blobs
    for k, v in forward_args.iteritems():
        if DEBUG:
            print 'shape of %s is ' % k
            print v.shape
        recurrent_net.blobs[k].reshape(*(v.shape))

    recurrent_net.forward(**forward_args)
    forward_args['cont_sentence'][:] = 1

    input_sentence = np.zeros((1,sample_n)) # start with EOS    
    embed_net.blobs['input_sentence'].reshape(1, sample_n) 
    
    for step in xrange(max_timestep):
        
        embed_out = embed_net.forward(input_sentence=input_sentence)
        forward_args['input_features'] = embed_out['embedded_sentence']
        # another lstm for global features
        if 'global_features' in recurrent_net.blobs:
            forward_args['global_features'] =  embed_out['embedded_sentence']
        blobs_out = recurrent_net.forward(**forward_args)

        word_scores = recurrent_net.blobs['predict'].data.copy()
        word_probs = blobs_out['probs'].copy()
        bbox_pred = blobs_out['bbox_pred'] if pred_bbox else None
        #suppress <unk> tag
        #word_probs[:,:,1] = 0
        sampled_words = np.zeros(sample_n, dtype=np.int32)
        for sample_id in xrange(sample_n):
            sampled_words[sample_id] = random_choice_from_probs(word_scores[0,sample_id,:], temp=temperature)
        finish_n = 0
        for i, w in zip(range(sample_n), sampled_words):
            if not pred_captions[i]:
                pred_captions[i] = [w]
                pred_logprobs[i] = math.log(max(word_probs[0,i,w],eps))
            elif pred_captions[i][-1] != 0:
                pred_captions[i].append(w)
                pred_logprobs[i] += math.log(max(word_probs[0,i,w],eps))
                pred_bbox_offsets[i,:] = bbox_pred[0,i,:] if pred_bbox else 0
            else:
                finish_n += 1
        input_sentence[:] = sampled_words
        forward_args['cont_sentence'][:] = 1
        if finish_n == sample_n:
            break
    if dedup:
        #deduplication
        pred_captions = [tuple(cap) for cap in pred_captions]
        pred_captions_set = set()
        pred_bbox_offsets_dedup = []
        pred_logprobs_dedup = []
        pred_captions_dedup = []
        for cap, offset, logprob in zip(pred_captions,pred_bbox_offsets, pred_logprobs):
            if cap not in pred_captions_set:
                pred_captions_set.add(cap)
                pred_captions_dedup.append(cap)
                pred_logprobs_dedup.append(logprob)
                pred_bbox_offsets_dedup.append(offset)
        pred_captions, pred_bbox_offsets, pred_logprobs = pred_captions_dedup, np.array(pred_bbox_offsets_dedup), pred_logprobs_dedup

    return pred_captions, pred_bbox_offsets, pred_logprobs
# multiple predictions for one region proposal
def region_captioning(feature_net, embed_net, recurrent_net, im, box, strategy):
    """Detect object classes in an image given object proposals.

    Arguments:
        feature_net (caffe.Net): CNN model for extracting features
        recurrent_net (caffe.Net): Recurrent model for generating captions and locations
        im (ndarray): color image to test (in BGR order)
        box (ndarray): 1 x 4 array of object proposal

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
    # for bbox unnormalization
    # TODO: put it in a more organized way
    bbox_mean = np.array([0,0,0,0]).reshape((1,4))
    bbox_stds = np.array([0.1, 0.1, 0.2, 0.2]).reshape((1,4))

    blobs, im_scales = _get_blobs(im, box)

    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs['data']
    rois = blobs['rois']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    # reshape network inputs
    feature_net.blobs['data'].reshape(*(blobs['data'].shape))
    feature_net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    feature_net.blobs['rois'].reshape(*(blobs['rois'].shape))

    feature_net.forward(data = im_blob, rois = blobs['rois'], im_info = blobs['im_info'])
    region_features = feature_net.blobs['region_features'].data.copy()
    #print region_features.shape
    assert(region_features.shape[1] == 1)
    
    # proposal boxes
    box_orig = rois[:, 1:5] / im_scales[0]
    # check bbox consistency
    assert(np.linalg.norm(box_orig - box) < 0.0001)
    
    if strategy['algorithm'] == 'sample':
        sample_n = strategy['samples'] 
        temperature = strategy['temperature']
    else:
        sample_n = 1
    feat_args = {'input_features': np.tile(region_features,(1, sample_n, 1))}
    opt_args = {}
    # global feature as an optional input: context
    if 'global_features' in feature_net.blobs:
        #changed according to the global feature shape
        opt_args['global_features'] = np.tile(feature_net.blobs['global_features'].data, (1,sample_n,1)) 
    
    
    bbox_pred_direct = ('bbox_pred' in feature_net.blobs)

    if bbox_pred_direct:
        # do greedy search
        if strategy['algorithm'] == 'sample':
            captions, _, logprobs = _multi_sampling(embed_net, recurrent_net, feat_args, opt_args, sample_n, \
                temperature=temperature, pred_bbox = False)
        else:
            captions, _, logprobs, _ = _greedy_search(embed_net, recurrent_net, feat_args, opt_args,\
             pred_bbox = False)
        #bbox target unnormalization
        box_offsets = np.tile(feature_net.blobs['bbox_pred'].data, (sample_n, 1))
    else:
        if strategy['algorithm'] == 'sample':

            captions, box_offsets, logprobs = _multi_sampling(embed_net, recurrent_net, feat_args, opt_args, \
                sample_n, temperature=temperature, pred_bbox = True)
        else:
            captions, box_offsets, logprobs, box_offsets_all_timesteps = _greedy_search(embed_net, recurrent_net, feat_args, opt_args, \
                pred_bbox = True)
   
    if strategy['algorithm'] == 'greedy' and not bbox_pred_direct:
        # return bbox prediction in all timesteps
        #bbox target unnormalization
        box_deltas = box_offsets_all_timesteps * bbox_stds + bbox_mean

        #do the transformation
        pred_boxes_all_timesteps = bbox_transform_inv(np.tile(box,(box_offsets_all_timesteps.shape[0],1)), box_deltas)
        pred_boxes_all_timesteps = clip_boxes(pred_boxes_all_timesteps, im.shape)
    else:
        pred_boxes_all_timesteps = None

    #bbox target unnormalization
    box_deltas = box_offsets * bbox_stds + bbox_mean

    #do the transformation
    pred_boxes = bbox_transform_inv(np.tile(box,(box_offsets.shape[0],1)), box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)
    
    return pred_boxes, captions, pred_boxes_all_timesteps
        
    

def sentence(vocab, vocab_indices):
    # consider <eos> tag with id 0 in vocabulary
    sentence = ' '.join([vocab[i] for i in vocab_indices])
    suffix = ' ' + vocab[0]
    if sentence.endswith(suffix):
      sentence = sentence[:-len(suffix)]
    return sentence

def softmax(softmax_inputs, temp):
  shifted_inputs = softmax_inputs - softmax_inputs.max()
  exp_outputs = np.exp(1. / temp * shifted_inputs)
  exp_outputs_sum = exp_outputs.sum()
  if math.isnan(exp_outputs_sum):
    return exp_outputs * float('nan')
  assert exp_outputs_sum > 0
  if math.isinf(exp_outputs_sum):
    return np.zeros_like(exp_outputs)
  eps_sum = 1e-20
  return exp_outputs / max(exp_outputs_sum, eps_sum)

def random_choice_from_probs(softmax_inputs, temp=1):

  probs = softmax(softmax_inputs, temp)
  r = random.random()
  
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 0  # return eos?

