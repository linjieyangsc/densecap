#!/usr/bin/env python
import itertools
import os
import re
import sys
import json
import time
import numpy as np
from collections import Counter 
VG_VERSION='1.0'
VG_PATH = '/home/ljyang/work/data/visual_genome'
VG_IMAGE_ROOT = '%s/images' % VG_PATH
VG_REGION_PATH = '%s/%s/region_descriptions.json' % (VG_PATH,VG_VERSION)
VG_METADATA_PATH = '%s/%s/image_data.json' % (VG_PATH,VG_VERSION)
vocabulary_size = 10000#10497#from dense caption paper

# punctuations to be removed
punct_list = ['.','?','!']
OUTPUT_DIR = 'data/visual_genome/%s' % VG_VERSION

# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = '<unk>'

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

def split_sentence(sentence):
	# break sentence into a list of words and punctuation
	sentence = [s.lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]
	return sentence

MAX_WORDS = 10

class VGDataProcessor:
	def __init__(self, image_regions, image_data, vocab=None,
			split_ids=[], max_words=MAX_WORDS):
		self.max_words = max_words
		num_empty_lines = 0
		self.images = []
		num_total = 0
		num_missing = 0
		num_captions = 0
		#known_images = {}
		phrases_all = []
		self.images = {}
		num_invalid_bbox = 0
		num_bbox = 0
		num_empty_phrase = 0
		for item,image_info in zip(image_regions, image_data):
			im_id = item['id']
			if im_id != image_info['id']:
				print 'region and image metadata inconsistent'
				exit()
			if not im_id in split_ids:
				continue
			#tokenize phrase
			num_bbox += len(item['regions'])
			regions_filt = []
			for obj in item['regions']:
				# remove invalid regions
				if obj['width'] == 0 or obj['height'] ==0:
					num_invalid_bbox += 1
					continue
				phrase  = obj['phrase'].strip().encode('ascii','ignore').lower()

				# remove empty sentence 
				if (len(phrase)==0):
					num_empty_phrase += 1
					continue
				# remove punctuation
				if phrase[-1] in punct_list:
					phrase = phrase[:-1]
				obj['phrase_tokens'] = split_sentence(phrase)
				# remove regions with caption longer than max_words
				if len(obj['phrase_tokens']) > max_words:
					continue
				regions_filt.append(obj)
				phrases_all.append(obj['phrase_tokens'])
			im_path = '%s/%d.jpg' % (VG_IMAGE_ROOT, im_id)
			self.images[im_id] = {'path': im_path, 'regions':regions_filt,
			'height':image_info['height'], 'width': image_info['width']}
		print "there are %d invalid bboxes out of %d" % (num_invalid_bbox, num_bbox)
		print "there are %d empty phrases after triming" % num_empty_phrase
		if vocab is None:
			self.init_vocabulary(phrases_all)
		else:
			self.vocabulary_inverted = vocab
		self.vocabulary = {}
		for index, word in enumerate(self.vocabulary_inverted):
			self.vocabulary[word] = index

	def init_vocabulary(self, phrases_all):
		words_to_count = {}
		word_freq = Counter(itertools.chain(*phrases_all))
		print "Found %d unique word tokens." % len(word_freq.items())
		vocab_freq = word_freq.most_common(vocabulary_size-1)
		self.vocabulary_inverted = [x[0] for x in vocab_freq]
		self.vocabulary_inverted.insert(0,UNK_IDENTIFIER)
		print "Using vocabulary size %d." % vocabulary_size
		print "The least frequent word in our vocabulary is '%s' and appeared %d times." % \
		(vocab_freq[-1][0], vocab_freq[-1][1])

	def dump_vocabulary(self, vocab_filename):
		print 'Dumping vocabulary to file: %s' % vocab_filename
		with open(vocab_filename, 'wb') as vocab_file:
			for word in self.vocabulary_inverted:
				vocab_file.write('%s\n' % word)
		print 'Done.'


VG_IMAGE_PATTERN = '%s/%%d.jpg' % VG_IMAGE_ROOT


SPLITS_PATTERN = VG_PATH + '/densecap_splits/%s.txt'

def process_dataset(split_name, vocab=None):
	split_image_ids = []
	with open(SPLITS_PATTERN % split_name, 'r') as split_file:
		for line in split_file.readlines():
			line_id = int(line.strip())
			split_image_ids.append(line_id)
	print 'split image number: %d' % len(split_image_ids)
	output_dataset_name = split_name
	print 'start loading json files...'
	t1 = time.time()
	regions_all = json.load(open(VG_REGION_PATH))
	image_data = json.load(open(VG_METADATA_PATH))
	t2 = time.time()
	print '%f seconds for loading' % (t2-t1)
	processor = VGDataProcessor(regions_all, image_data,
			split_ids=split_image_ids, vocab=vocab)
	
	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)
	if vocab is None:
		vocab_out_path = '%s/vocabulary.txt' % OUTPUT_DIR
		processor.dump_vocabulary(vocab_out_path)
	#dump image region dict
	with open(OUTPUT_DIR+'/%s_gt_regions.json' % split_name, 'w') as f:
		json.dump(processor.images,f)

	return processor.vocabulary_inverted

def process_vg():
	vocab = None

	datasets = ['train', 'val', 'test']
			
	for split_name in datasets:
		vocab = process_dataset(split_name, vocab=vocab)

if __name__ == "__main__":
	process_vg()
