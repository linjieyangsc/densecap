import os
import nltk
import itertools
import cPickle
from hashlib import sha1
import random
import sys
import time
random.seed(3)

sys.path.append('./examples/coco_caption/')
from hdf5_sequence_generator import SequenceGenerator, HDF5SequenceWriter
vocabulary_size = 20000
UNK_IDENTIFIER = "<unk>"#follow the coco routine

			      
MAX_WORDS = 30
MAX_HASH = 100000

class BooksSequenceGenerator(SequenceGenerator):
	def __init__(self, batch_num_streams, sentences, vocab = None, 
			max_words = MAX_WORDS):
		self.max_words = max_words
		#self.images = images
		# Tokenize the sentences into words
		tokenized_sentences = [sent.split() for sent in sentences]
		print ' %d sentences in total' % len(tokenized_sentences) 
		# Count the word frequencies
		#word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
		#print "Found %d unique words tokens." % len(word_freq.items())
		self.vocabulary_inverted = vocab
		
		self.sentences = tokenized_sentences
		self.vocabulary = dict([(w,i) for i,w in enumerate(self.vocabulary_inverted)])
		#self.image_sentence_pairs = zip(images,tokenized_sentences)
		self.index = 0
		self.num_resets = 0
		self.num_truncates = 0
		self.num_pads = 0
		self.num_outs =0
		#self.image_list=[]
		self.pad = False
		self.truncate = False
		self.align = False
		SequenceGenerator.__init__(self)
		self.batch_num_streams = batch_num_streams
		self.negative_one_padded_streams = frozenset(('input_sentence','target_sentence'))

	def streams_exhausted(self):
		return self.num_resets > 0
	
	def next_line(self):
		num_lines = float(len(self.sentences))
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
		line = self.sentences[self.index]
		stream = self.line_to_stream(line)
		pad = self.max_words - (len(stream) + 1) if self.pad else 0
		if pad > 0: self.num_pads += 1
		self.num_outs += 1
		out = {}
		out['stage_indicators'] = [1] * (len(stream) + 1) + [0] * pad
		out['cont_sentence'] = [0] + [1] * len(stream) + [0] * pad
		out['input_sentence'] = [0] + stream + [-1] * pad
		out['target_sentence'] = stream + [0] + [-1] * pad
		truncated = False
		if self.truncate:
			for key, val in out.iteritems():
				if len(val) > self.max_words:
					out[key] = val[:self.max_words]
					truncated = True
			self.num_truncates += truncated
		#image_hash = self.image_hash(image_filename)
		#out['hashed_image_path'] = [image_hash] * len(out['input_sentence'])
		#self.image_list.append((image_filename, image_hash))
		self.next_line()
		return out

	def image_hash(self, filename):
		image_hash = int(sha1(filename).hexdigest(), 16) % MAX_HASH
		assert image_hash == float(image_hash)
		return image_hash


#
BUFFER_SIZE=100
OUTPUT_DIR = './models/lstm/h5_data_lm/buffer_%d_vocab_%d' % (BUFFER_SIZE, vocabulary_size)
SPLITS_PATTERN = '/home/a-linjieyang/work/skip-thoughts/training/sents_%s.txt'
#SPLITS_CAP_PATTERN = '/home/a-linjieyang/work/video_caption/dreamstime/%s_list_cap_distill.txt'
OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR
VOCAB_PATH = '/home/a-linjieyang/work/skip-thoughts/training/vocabulary'
def process_dataset(split_name, batch_stream_length, vocab=None):
	print 'loading dataset %s ...' % split_name
	t1 = time.time()
	with open(SPLITS_PATTERN % split_name, 'r') as split_file:
		split_sentences = [line.strip() for line in split_file]
	#if len(split_sentences) != batch_stream_length:
	#	print 'sentence number not match! %d vs %d' % (len(split_sentences),batch_stream_length)
	#	exit()
	t2 = time.time()
	print 'loaded, %s seconds elapsed' % (t2-t1)
	#with open(SPLITS_CAP_PATTERN % split_name,'r') as split_cap_file:
	#	split_sentences = [line[:-1] for line in split_cap_file]
	output_dataset_name = '%s_%d' % (split_name, MAX_WORDS)
	output_path = OUTPUT_DIR_PATTERN % output_dataset_name
	print 'initializing seq generator ...'
	sg = BooksSequenceGenerator(BUFFER_SIZE, split_sentences, vocab=vocab)
	t3 = time.time()
	print 'finished. %s seconds elapsed' % (t3-t2)
	split_sentences = None
	sg.batch_stream_length = batch_stream_length
	writer = HDF5SequenceWriter(sg, output_dir=output_path)
	#t4 = time.time()
	
	writer.write_to_exhaustion()
	t4 = time.time()
	print 'saving hdf5 files finished. %s seconds elapsed' % (t4-t3)
	writer.write_filelists()
	return sg.vocabulary_inverted
def process_books():
	vocab=[]
	#vocab.append('<eos>')
	vocab.append(UNK_IDENTIFIER)
	vocab_full = cPickle.load(open(VOCAB_PATH))
	for kk,vv in vocab_full.iteritems():
		if vv > vocabulary_size:
			break
		vocab.append(kk)
	#vocab = vocab_full[:vocabulary_size]
	#write vocabulary
	with open(VOCAB_PATH+'.txt','w') as f:
		for w in vocab:
			f.write(w+'\n')
	exit()
	print vocab[:3]
	print vocab[-1]
	print len(vocab)
	datasets = [
	('train', 1000000),# 70304016),
	('val', 400000), #3700212),
	]
	for split_name, batch_stream_length in datasets:
		process_dataset(split_name, batch_stream_length,
			vocab = vocab)
if __name__ == "__main__":
	process_books()
