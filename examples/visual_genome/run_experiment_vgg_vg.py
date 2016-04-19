#!/usr/bin/env python

from collections import OrderedDict
import json
import numpy as np
import pprint
import cPickle as pickle
import string
import sys
import time
# seed the RNG so we evaluate on the same subset each time
np.random.seed(seed=0)
sys.path.append('./examples/coco_caption/')  
from vg_to_hdf5_data import *
from region_captioner import RegionCaptioner
COCO_EVAL_PATH = '/media/researchshare/linjie/data/MS_COCO/coco-caption/'
sys.path.append(COCO_EVAL_PATH)
from pycocoevalcap.vg_eval import VgEvalCap

class CaptionExperiment():
  # captioner is an initialized Captioner (captioner.py)
  # dataset is a dict: image path -> [caption1, caption2, ...]
  def __init__(self, captioner, dataset, dataset_cache_dir, cache_dir, sg):
    self.captioner = captioner
    self.sg = sg
    self.dataset_cache_dir = dataset_cache_dir
    self.cache_dir = cache_dir
    for d in [dataset_cache_dir, cache_dir]:
      if not os.path.exists(d): os.makedirs(d)
    self.dataset = dataset
    self.images = dataset.keys()
    self.init_caption_list(dataset)
    self.caption_scores = [None] * len(self.images)
    print 'Initialized caption experiment: %d images, %d captions' % \
        (len(self.images), len(self.captions))

  def init_caption_list(self, dataset):
    self.captions = []
    for image, captions in dataset.iteritems():
      for cap_stream, cap, loc in captions:
        self.captions.append({'source_image': image, 'caption': cap_stream, 'location': loc})
    # Sort by length for performance.
    self.captions.sort(key=lambda c: len(c['caption']))

  def compute_descriptors(self):
    descriptor_filename = '%s/descriptors.npz' % self.dataset_cache_dir
    if os.path.exists(descriptor_filename):
      self.descriptors = np.load(descriptor_filename)['descriptors']
    else:
      self.descriptors = self.captioner.compute_descriptors(self.images,output_name='conv6')
      np.savez_compressed(descriptor_filename, descriptors=self.descriptors)

  def score_captions(self, image_index, output_name='probs'):
    assert image_index < len(self.images)
    caption_scores_dir = '%s/caption_scores' % self.cache_dir
    if not os.path.exists(caption_scores_dir):
      os.makedirs(caption_scores_dir)
    caption_scores_filename = '%s/scores_image_%06d.pkl' % \
        (caption_scores_dir, image_index)
    if os.path.exists(caption_scores_filename):
      with open(caption_scores_filename, 'rb') as caption_scores_file:
        outputs = pickle.load(caption_scores_file)
    else:
      outputs = self.captioner.score_captions(self.descriptors[image_index],
          self.captions, output_name=output_name, caption_source='gt',
          verbose=False)
      self.caption_stats(image_index, outputs)
      with open(caption_scores_filename, 'wb') as caption_scores_file:
        pickle.dump(outputs, caption_scores_file)
    self.caption_scores[image_index] = outputs

  def caption_stats(self, image_index, caption_scores):
    image_path = self.images[image_index]
    for caption, score in zip(self.captions, caption_scores):
      assert caption['caption'] == score['caption']
      score['stats'] = gen_stats(score['prob'])
      score['correct'] = (image_path == caption['source_image'])

  def normalize_caption_scores(self, caption_index, stats=['log_p', 'log_p_word']):
    scores = [s[caption_index] for s in self.caption_scores]
    for stat in stats:
      log_stat_scores = np.array([score['stats'][stat] for score in scores])
      stat_scores = np.exp(log_stat_scores)
      mean_stat_score = np.mean(stat_scores)
      log_mean_stat_score = np.log(mean_stat_score)
      for log_stat_score, score in zip(log_stat_scores, scores):
        score['stats']['normalized_' + stat] = log_stat_score - log_mean_stat_score

  def generation_experiment(self, strategy, max_batch_size=1000):
    # Compute image descriptors.
    print 'Computing image descriptors'
    self.compute_descriptors()

    do_batches = (strategy['type'] == 'beam' and strategy['beam_size'] == 1) or \
        (strategy['type'] == 'sample' and
         ('temp' not in strategy or strategy['temp'] in (1, float('inf'))) and
         ('num' not in strategy or strategy['num'] == 1))

    num_images = len(self.images)
    batch_size = min(max_batch_size, num_images) if do_batches else 1
    t1 =time.time()
    # Generate captions for all images.
    all_results = [None] * num_images
    for image_index in xrange(0, num_images, batch_size):
      batch_end_index = min(image_index + batch_size, num_images)
      sys.stdout.write("\rGenerating captions for image %d/%d" %
                       (image_index, num_images))
      sys.stdout.flush()
      for batch_image_index in xrange(image_index, batch_end_index):
          captions, caption_probs, locations = self.captioner.predict_caption(
              self.descriptors[batch_image_index], strategy=strategy)
          #best_caption, best_loc, max_log_prob = None, None
          #for caption, probs in zip(captions, caption_probs):
          print 'mean of descriptor is %f' % np.mean(self.descriptors[batch_image_index])


	  #exit()
          
          locations = [get_bbox_coord(location_seq) for location_seq in locations]
          log_probs = [gen_stats(probs)['log_p_word'] for probs in caption_probs]  
          #print log_probs
          print 'locations of first and last caption'
          print locations[0]
          print locations[-1]
          result = []
          for cap, log_prob, location in zip(captions, log_probs, locations):
            result.append({'caption':cap,'log_prob':log_prob,'location':location})        

          result_nms = nms(result)
          nms_captions = [x['caption'] for x in result_nms]
          print 'there are %d captions after nms' % len(result_nms)
          print nms_captions
          all_results[batch_image_index] = result_nms
    sys.stdout.write('\n')
    t2 = time.time()
    print "%f seconds elapsed" % (t2-t1)
    
    image_ids = range(len(self.images))#dummy index
    # Collect model/reference captions, formatting the model's captions and
    # each set of reference captions as a list of len(self.images) strings.
    exp_dir = '%s/generation' % self.cache_dir
    if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
    # For each image, write out the highest probability caption.
    model_caption_locations = [None] * len(self.images)
    reference_caption_locations = [None] * len(self.images)
    for image_index, image in enumerate(self.images):
      model_caption_locations[image_index] = []
      for x in all_results[image_index]:
        anno = {'image_id':image_ids[image_index], 'caption': self.captioner.sentence(x['caption']), 
	      'location_seq': x['location'].tolist(), 'location': x['location'][-1,:].tolist()}
        model_caption_locations[image_index].append(anno) 
      
      new_ref_caption_locations = []
      for reference_index, (_, phrase, location) in enumerate(self.dataset[image]):
        phrase = ' '.join(phrase)
        location = get_bbox_coord(np.array(location).reshape(1,4)).reshape(4)
        anno={'image_id':image_ids[image_index], 'caption': phrase, 'location': location}
        new_ref_caption_locations.append(anno) 
      #merge regions with large overlapped areas
      reference_caption_locations[image_index] = gt_region_merge(new_ref_caption_locations)
      print '%d regions after merging' % len(reference_caption_locations[image_index])
   
    generation_result = [{
      'image_id': image_ids[image_index],
      'image_path': image_path,
      'caption_locations': model_caption_locations[image_index]
    } for (image_index, image_path) in enumerate(self.images)]
    json_filename = '%s/generation_result.json' % self.cache_dir
    print 'Dumping result to file: %s' % json_filename
    with open(json_filename, 'w') as json_file:
      json.dump(generation_result, json_file)
    #generation_result = self.sg.coco.loadRes(json_filename)
    vg_evaluator = VgEvalCap(reference_caption_locations, model_caption_locations)
    vg_evaluator.params['image_id'] = image_ids
    vg_evaluator.evaluate()
def get_bbox_coord(norm_coord, do_clip=True):
  #input is a nx4 numpy array in normalized bbox coordinates
	#print norm_coord.shape
	#print norm_coord
	bboxes_coord = np.zeros(norm_coord.shape)
	#x,y,w,h
	bboxes_coord[:, :2] = norm_coord[:, :2]+0.5
	bboxes_coord[:, 2:] = np.exp(norm_coord[:, 2:])
	
	#x1,y1,x2,y2
	bboxes_coord2 = np.zeros(norm_coord.shape)
	bboxes_coord2[:, :2] = bboxes_coord[:, :2] - bboxes_coord[:, 2:] * 0.5
	bboxes_coord2[:, 2:] = bboxes_coord[:, :2] + bboxes_coord[:, 2:] * 0.5
	#clipping all coordinates to [0,1]
	if do_clip:
		bboxes_coord2 = np.minimum(np.maximum(bboxes_coord2, 0), 1)
	return bboxes_coord2
	
def nms(region_info, bbox_th=0.3):
	#non-maximum surpression
  region_info.sort(key = lambda x: -x['log_prob'])
  #keep_index = []
  region_n = len(region_info)
  #fast computation of pairwise IoU
  #pick the bbox of last timestep of each sample
  #print 'region_info length %d' % len(region_info)
  all_bboxes = np.array([x['location'][-1,:] for x in region_info])# nx4 matrix
  #print 'bboxes shape: '
  #print all_bboxes.shape
  bboxes_coord2 = all_bboxes	
  #area, intersection area, union area
  bbox_areas = (bboxes_coord2[:,2] - bboxes_coord2[:,0]) * \
    (bboxes_coord2[:, 3] - bboxes_coord2[:, 1])
  print 'bbox areas'
  print bbox_areas
  x_a1 = bboxes_coord2[:,0].reshape(region_n,1)
  x_a2 = bboxes_coord2[:,2].reshape(region_n,1)
  x_b1 = bboxes_coord2[:,0].reshape(1,region_n)
  x_b2 = bboxes_coord2[:,2].reshape(1,region_n)
  y_a1 = bboxes_coord2[:,1].reshape(region_n,1)
  y_a2 = bboxes_coord2[:,3].reshape(region_n,1)
  y_b1 = bboxes_coord2[:,1].reshape(1,region_n)
  y_b2 = bboxes_coord2[:,3].reshape(1,region_n)
  bbox_pair_x_diff = np.maximum(0, np.minimum(x_a2, x_b2) - np.maximum(x_a1, x_b1))
  bbox_pair_y_diff = np.maximum(0, np.minimum(y_a2, y_b2) - np.maximum(y_a1, y_b1))
  inter_areas = bbox_pair_x_diff * bbox_pair_y_diff
  print 'inter areas'
  print inter_areas
  #IoU
  union_areas = bbox_areas.reshape(region_n,1) + bbox_areas.reshape(1,region_n)
  print 'union_areas'
  print union_areas
  bbox_iou = inter_areas / (union_areas - inter_areas)
  print 'bbox iou'
  print bbox_iou
  bbox_iou_th = bbox_iou < bbox_th
  keep_flag = np.ones((region_n),dtype=np.uint8)

  for i in xrange(region_n-1):
    if keep_flag[i]:
      keep_flag[i+1:] = np.logical_and(keep_flag[i+1:], bbox_iou_th[i,i+1:])  
  print 'sum of keep flag'
  print keep_flag.sum()
  return [region_info[i] for i in xrange(region_n) if keep_flag[i]]	

def gt_region_merge(region_info, bbox_th=0.7):
  #merging ground truth bboxes

  #keep_index = []
  region_n = len(region_info)
  region_merged = []
  #fast computation of pairwise IoU
  #pick the bbox of last timestep of each sample
  all_bboxes = np.array([x['location'] for x in region_info])# nx4 matrix
  print all_bboxes.shape
  bboxes_coord2 = all_bboxes  
  #area, intersection area, union area
  bbox_areas = (bboxes_coord2[:,2] - bboxes_coord2[:,0]) * \
    (bboxes_coord2[:, 3] - bboxes_coord2[:, 1])

  x_a1 = bboxes_coord2[:,0].reshape(region_n,1)
  x_a2 = bboxes_coord2[:,2].reshape(region_n,1)
  x_b1 = bboxes_coord2[:,0].reshape(1,region_n)
  x_b2 = bboxes_coord2[:,2].reshape(1,region_n)
  y_a1 = bboxes_coord2[:,1].reshape(region_n,1)
  y_a2 = bboxes_coord2[:,3].reshape(region_n,1)
  y_b1 = bboxes_coord2[:,1].reshape(1,region_n)
  y_b2 = bboxes_coord2[:,3].reshape(1,region_n)
  bbox_pair_x_diff = np.maximum(0, np.minimum(x_a2, x_b2) - np.maximum(x_a1, x_b1))
  bbox_pair_y_diff = np.maximum(0, np.minimum(y_a2, y_b2) - np.maximum(y_a1, y_b1))
  inter_areas = bbox_pair_x_diff * bbox_pair_y_diff
  
  #IoU
  union_areas = bbox_areas.reshape(region_n,1) + bbox_areas.reshape(1,region_n)
  
  bbox_iou = inter_areas / (union_areas - inter_areas)
  bbox_iou_th = bbox_iou > bbox_th
  bbox_iou_overlap_n = bbox_iou_th.sum(axis = 0)

  merge_flag = np.ones((region_n),dtype=np.uint8)
  unmerged_region = region_n
  while unmerged_region > 0:
    max_overlap_id = np.argmax(bbox_iou_overlap_n)
    assert bbox_iou_overlap_n[max_overlap_id] > 0
    merge_group = np.nonzero(bbox_iou_th[max_overlap_id,:] & merge_flag)[0]
    unmerged_region -= len(merge_group)
    merge_flag[merge_group] = 0
    bbox_iou_overlap_n[merge_group] = 0
    temp = bboxes_coord2[merge_group,:]
    print temp.shape
    print merge_group
    bbox_group = bboxes_coord2[merge_group,:].reshape(len(merge_group),4)
    caption_group = [region_info[i]['caption'] for i in merge_group]
    bbox_mean = np.mean(bbox_group, axis = 0).tolist()
    region_merged.append({'image_id':region_info[max_overlap_id]['image_id'], \
      'captions': caption_group, 'location': bbox_mean})
  return region_merged    

def gen_stats(prob):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += np.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  try:
    stats['perplex'] = np.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = np.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  return stats

def main():
  MAX_IMAGES = 10  # -1 to use all images
  TAG = 'vg_2layer_factored'
  if MAX_IMAGES >= 0:
    TAG += '_%dimages' % MAX_IMAGES
  eval_on_test = False
  
  ITER = 150000
  MODEL_FILENAME = 'dense_cap_cross3_iter_%d' % ITER
  DATASET_NAME = 'vg_test'

  TAG += '_%s' % DATASET_NAME
  MODEL_DIR = './examples/visual_genome'
  MODEL_FILE = '%s/%s.caffemodel' % (MODEL_DIR, MODEL_FILENAME)
  IMAGE_NET_FILE = './examples/visual_genome/vgg_deploy.prototxt'
  LSTM_NET_FILE = './examples/visual_genome/joint_pred_cross3.deploy.prototxt'#joint_pred_cross.deploy.prototxt for cross version
  NET_TAG = '%s_%s' % (TAG, MODEL_FILENAME)
  DATASET_SUBDIR = '%s/%s_ims' % (DATASET_NAME,
      str(MAX_IMAGES) if MAX_IMAGES >= 0 else 'all')
  DATASET_CACHE_DIR = './retrieval_cache/%s/%s' % (DATASET_SUBDIR, MODEL_FILENAME)
  VOCAB_FILE = './examples/visual_genome/h5_data_distill/buffer_100/vocabulary.txt'
  DEVICE_ID = 4
  with open(VOCAB_FILE, 'r') as vocab_file:
    vocab = [line.strip() for line in vocab_file.readlines()]
  #coco = COCO(COCO_ANNO_PATH % DATASET_NAME)
  #image_root = '/media/researchshare/linjie/data/dreamstime/images'#COCO_IMAGE_PATTERN % DATASET_NAME
  eval_image_file = '/media/researchshare/linjie/data/visual-genome/densecap_splits/test.txt'
  #eval_caption_file = '/home/a-linjieyang/work/video_caption/dreamstime/val_list_cap.txt'
  with open(eval_image_file, 'r') as split_file:
    split_image_ids = [int(line.strip()) for line in split_file]
  #split_image_ids = [2342728]
  vg_sample_region_path = '/media/researchshare/linjie/data/visual-genome/sample_region_descriptions.json'
  vg_sample_meta_path = '/media/researchshare/linjie/data/visual-genome/sample_image_data.json'
  #regions_all = json.load(open(vg_sample_region_path))
  #image_data = json.load(open(vg_sample_meta_path))
  regions_all = json.load(open(VG_REGION_PATH))
  image_data = json.load(open(VG_METADATA_PATH))
  #print image_data
  print 'region data loaded.'
  #with open(eval_caption_file, 'r') as split_cap_file:
  #  split_sentences = [line.strip() for line in split_cap_file]
  sg = VGSequenceGenerator(BUFFER_SIZE, regions_all, image_data, split_ids=split_image_ids, vocab=vocab, align=True)
  dataset = {}
  for image_path, phrase, coord in sg.image_phrase_pairs:#[impath,phrase_tokens, norm_coord]
    if image_path not in dataset:
      dataset[image_path] = []

    dataset[image_path].append((sg.line_to_stream(phrase),phrase,coord))
    #dataset[image_path].append((sg.line_to_stream(sentence), sentence))
  print 'Original dataset contains %d images' % len(dataset.keys())
  if 0 <= MAX_IMAGES < len(dataset.keys()):
    all_keys = dataset.keys()
    perm = np.random.permutation(len(all_keys))[:MAX_IMAGES]
    chosen_keys = set([all_keys[p] for p in perm])
    for key in all_keys:
      if key not in chosen_keys:
        del dataset[key]
    print 'Reduced dataset to %d images' % len(dataset.keys())
  if MAX_IMAGES < 0: MAX_IMAGES = len(dataset.keys())
  captioner = RegionCaptioner(MODEL_FILE, IMAGE_NET_FILE, LSTM_NET_FILE, VOCAB_FILE,
                        device_id=DEVICE_ID)
  beam_size = 30

  #generation_strategy = {'type': 'beam', 'beam_size': beam_size}
  generation_strategy = {'type':'sample', 'num':1000, 'temp': 1}
  if generation_strategy['type'] == 'beam':
    strategy_name = 'beam%d' % generation_strategy['beam_size']
  elif generation_strategy['type'] == 'sample':
    strategy_name = 'sample%0.1f' % generation_strategy['temp']
  else:
    raise Exception('Unknown generation strategy type: %s' % generation_strategy['type'])
  CACHE_DIR = '%s/%s' % (DATASET_CACHE_DIR, strategy_name)
  experimenter = CaptionExperiment(captioner, dataset, DATASET_CACHE_DIR, CACHE_DIR, sg)
  captioner.set_image_batch_size(min(100, MAX_IMAGES))
  
  experimenter.generation_experiment(generation_strategy)
  captioner.set_caption_batch_size(min(MAX_IMAGES * 5, 1000))
  #experimenter.retrieval_experiment()

if __name__ == "__main__":
  main()
