#!/usr/bin/env python
import nltk
import itertools
from hashlib import sha1
import os
import random
random.seed(3)
import re
import sys
import json
sys.path.append('./examples/coco_caption/')
import time
import numpy as np
VG_PATH = '/media/researchshare/linjie/data/visual-genome'
#COCO_TOOL_PATH = '%s/coco/PythonAPI/' % COCO_PATH
VG_IMAGE_ROOT = '%s/images' % VG_PATH
VG_REGION_PATH = '%s/region_descriptions.json' % VG_PATH
VG_METADATA_PATH = '%s/image_data.json' % VG_PATH
MAX_HASH = 100000
vocabulary_size = 10000#10497#from dense caption paper
#sys.path.append(COCO_TOOL_PATH)
#from pycocotools.coco import COCO
strip_words = ['there is ','there are ', 'this seems to be ', 'it seems to be ', 
'it is ','that is ','this is ']
punct_list = ['.','?','!']
from hdf5_sequence_generator2 import SequenceGenerator, HDF5SequenceWriter

# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = '<unk>'

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def split_sentence(sentence):
	# break sentence into a list of words and punctuation
	sentence = [s.lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]
	# remove the '.' from the end of the sentence
	if sentence[-1] != '.':
		# print "Warning: sentence doesn't end with '.'; ends with: %s" % sentence[-1]
		return sentence
	return sentence[:-1]

MAX_WORDS = 11

class VGSequenceGenerator(SequenceGenerator):
	def __init__(self, batch_num_streams, image_regions, image_data, vocab=None,
			split_ids=[], max_words=MAX_WORDS, align=True, shuffle=True,
							 pad=True, truncate=True):
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
				#remove invalid regions
				if obj['width'] == 0 or obj['height'] ==0:
					num_invalid_bbox += 1
					continue
				phrase  = obj['phrase'].strip()
				#strip referring words
				for w in strip_words:
					if phrase[:len(w)] == w:
						phrase = phrase[len(w):]
				if (len(phrase)==0):
					num_empty_phrase += 1
					continue
				if phrase[-1] in punct_list:
					phrase = phrase[:-1]
				obj['phrase_tokens'] = nltk.word_tokenize(phrase.lower())
				#remove regions with caption longer than max_words
				if len(obj['phrase_tokens']) >= max_words:
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
		self.image_phrase_pairs = []
		num_no_sentences = 0
		for image_id, metadata in self.images.iteritems():
			if not metadata['regions']:
				num_no_sentences += 1
				print 'Warning (#%d): image with no sentences: %d' % (num_no_sentences, image_id)
			for region in metadata['regions']:
				norm_coord = self.get_normalized_coordinates(metadata['height'],metadata['width'],region)
				self.image_phrase_pairs.append((metadata['path'], region['phrase_tokens'], 
				norm_coord))
		self.index = 0
		self.num_resets = 0
		self.num_truncates = 0
		self.num_pads = 0
		self.num_outs = 0
		self.image_list = []
		SequenceGenerator.__init__(self)
		self.batch_num_streams = batch_num_streams
		# make the number of image/sentence pairs a multiple of the buffer size
		# so each timestep of each batch is useful and we can align the images
		if align:
			num_pairs = len(self.image_phrase_pairs)
			remainder = num_pairs % batch_num_streams
			if remainder > 0:
				num_needed = batch_num_streams - remainder
				for i in range(num_needed):
					choice = random.randint(0, num_pairs - 1)
					self.image_phrase_pairs.append(self.image_phrase_pairs[choice])
			assert len(self.image_phrase_pairs) % batch_num_streams == 0
		if shuffle:
			random.shuffle(self.image_phrase_pairs)
		self.pad = pad
		self.truncate = truncate
		self.negative_one_padded_streams = frozenset(('input_sentence', 'target_sentence'))

	def streams_exhausted(self):
		return self.num_resets > 0

	def get_normalized_coordinates(self, im_height, im_width, region):
		x_b = region['x'] + region['width'] / 2
		y_b = region['y'] + region['height'] / 2
		x_o = im_width / 2
		y_o = im_height / 2
		w_b = region['width']
		h_b = region['height']
		if w_b==0 or h_b==0:
			print "invalid region size"
			print "w_b is %d, h_b is %d" %(w_b,h_b)
			exit()
		tx = float(x_b - x_o) / im_width
		ty = float(y_b - y_o) / im_height
		tw = np.log(float(w_b)/im_width)
		th = np.log(float(h_b)/im_height)
		return (tx,ty,tw,th)

	def init_vocabulary(self, phrases_all, min_count=15):
		words_to_count = {}
		word_freq = nltk.FreqDist(itertools.chain(*phrases_all))
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

	def dump_image_file(self, image_filename, dummy_image_filename=None):
		print 'Dumping image list to file: %s' % image_filename
		with open(image_filename, 'wb') as image_file:
			for image_path, _ in self.image_list:
				image_file.write('%s\n' % image_path)
		if dummy_image_filename is not None:
			print 'Dumping image list with dummy labels to file: %s' % dummy_image_filename
			with open(dummy_image_filename, 'wb') as image_file:
				for path_and_hash in self.image_list:
					image_file.write('%s %d\n' % path_and_hash)
		print 'Done.'

	def next_line(self):
		num_lines = float(len(self.image_phrase_pairs))
		self.index += 1
		if self.index == 1 or self.index == num_lines or self.index % 10000 == 0:
			print 'Processed %d/%d (%f%%) lines' % (self.index, num_lines,
							100 * self.index / num_lines)
		if self.index == num_lines:
			self.index = 0
			self.num_resets += 1

	def line_to_stream(self, sentence):
		stream = []
		for word in sentence:
			word = word.strip()
			if word in self.vocabulary:
				stream.append(self.vocabulary[word])
			else:  # unknown word; append UNK
				stream.append(self.vocabulary[UNK_IDENTIFIER])
		# increment the stream -- 0 will be the EOS character
		stream = [s + 1 for s in stream]
		return stream

	def get_pad_value(self, stream_name):
		return -1 if stream_name in self.negative_one_padded_streams else 0

	def get_streams(self):
		image_filename, line, coord = self.image_phrase_pairs[self.index]
		stream = self.line_to_stream(line)
		pad = self.max_words - (len(stream) + 1) if self.pad else 0
		if pad > 0: self.num_pads += 1
		self.num_outs += 1
		coord_arr = np.array(coord)#need reshape?
		coord_pad = np.zeros(4)
		out = {}
		out['stage_indicators'] = [1] * (len(stream) + 1) + [0] * pad
		out['cont_sentence'] = [0] + [1] * len(stream) + [0] * pad
		out['input_sentence'] = [0] + stream + [-1] * pad
		out['target_sentence'] = stream + [0] + [-1] * pad
		out['target_coord'] = [coord_pad] + [coord_arr] * len(stream) + [coord_pad] * pad
		#print "target coord length %d" % len(out['target_coord'])
		#exit()
		truncated = False
		if self.truncate:
			for key, val in out.iteritems():
				if len(val) > self.max_words:
					
					out[key] = val[:self.max_words]
					truncated = True
			self.num_truncates += truncated
			
		image_hash = self.image_hash(image_filename)
		out['hashed_image_path'] = [image_hash] * len(out['input_sentence'])
		self.image_list.append((image_filename, image_hash))
		self.next_line()
		return out

	def image_hash(self, filename):
		image_hash = int(sha1(filename).hexdigest(), 16) % MAX_HASH
		assert image_hash == float(image_hash)
		return image_hash

#COCO_ANNO_PATH = '%s/annotations/captions_%%s2014.json' % COCO_PATH
VG_IMAGE_PATTERN = '%s/%%d.jpg' % VG_IMAGE_ROOT

BUFFER_SIZE = 100
OUTPUT_DIR = './examples/visual_genome/h5_data_distill/buffer_%d' % BUFFER_SIZE
SPLITS_PATTERN = '/media/researchshare/linjie/data/visual-genome/densecap_splits/%s.txt'
OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR

def process_dataset(split_name, coco_split_name, batch_stream_length,
										vocab=None, aligned=True):
	split_image_ids = []
	with open(SPLITS_PATTERN % split_name, 'r') as split_file:
		for line in split_file.readlines():
			test_id = line.strip()[:-4]
			if test_id.isdigit():
				split_image_ids.append(int(test_id))
	print 'split image number: %d' % len(split_image_ids)
	output_dataset_name = split_name
	if aligned:
		output_dataset_name += '_aligned_%d' % MAX_WORDS
	else:
		output_dataset_name += '_unaligned'
	output_path = OUTPUT_DIR_PATTERN % output_dataset_name
	#coco = COCO(COCO_ANNO_PATH % coco_split_name)
	#image_root = VG_IMAGE_ROOT
	print 'start loading json files...'
	t1 = time.time()
	regions_all = json.load(open(VG_REGION_PATH))
	image_data = json.load(open(VG_METADATA_PATH))
	t2 = time.time()
	print '%f seconds for loading' % (t2-t1)
	sg = VGSequenceGenerator(BUFFER_SIZE, regions_all, image_data,
			split_ids=split_image_ids, vocab=vocab, align=aligned, pad=aligned,
			truncate=aligned)
	
	#early stop here
	#return sg.vocabulary_inverted
	
	sg.batch_stream_length = batch_stream_length
	writer = HDF5SequenceWriter(sg, output_dir=output_path)
	writer.write_to_exhaustion()
	writer.write_filelists()
	if vocab is None:
		vocab_out_path = '%s/vocabulary.txt' % OUTPUT_DIR
		sg.dump_vocabulary(vocab_out_path)
	image_out_path = '%s/image_list.txt' % output_path
	image_dummy_labels_out_path = '%s/image_list.with_dummy_labels.txt' % output_path
	sg.dump_image_file(image_out_path, image_dummy_labels_out_path)
	#dump image region dict
	with open(OUTPUT_DIR+'/%s_gt_regions.json' % split_name, 'w') as f:
		json.dump(sg.images,f)
	num_outs = sg.num_outs
	num_pads = sg.num_pads
	num_truncates = sg.num_truncates
	print 'Padded %d/%d sequences; truncated %d/%d sequences' % \
			(num_pads, num_outs, num_truncates, num_outs)
	return sg.vocabulary_inverted

def process_vg(include_trainval=False):
	vocab = None
	datasets = [
			('train', 'train', 100000, True),
			('val', 'val', 100000, True),
			('test', 'val', 100000, True),
		]
	# Also create a 'trainval' set if include_trainval is set.
	# ./data/coco/make_trainval.py must have been run for this to work.
	if include_trainval:
		datasets += [
			('trainval', 'trainval', 100000, True),
			]
	for split_name, coco_split_name, batch_stream_length, aligned in datasets:
		vocab = process_dataset(split_name, coco_split_name, batch_stream_length,
														vocab=vocab, aligned=aligned)

if __name__ == "__main__":
	process_vg(include_trainval=False)
