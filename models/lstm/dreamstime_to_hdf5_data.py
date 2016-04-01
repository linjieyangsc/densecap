import os
import nltk
import itertools
import cPickle
from hashlib import sha1
import random
import sys
random.seed(3)

sys.path.append('./examples/coco_caption/')
from hdf5_sequence_generator import SequenceGenerator, HDF5SequenceWriter
vocabulary_size = 10000
UNK_IDENTIFIER = "<unk>"#follow the coco routine

			      
MAX_WORDS = 20
MAX_HASH = 100000

class DtSequenceGenerator(SequenceGenerator):
	def __init__(self, batch_num_streams, images, sentences, vocab = None, 
			max_words = MAX_WORDS, align = True, pad=True, truncate=True):
		self.max_words = max_words
		self.images = images
		# Tokenize the sentences into words
		tokenized_sentences = [nltk.word_tokenize(sent.lower()) for sent in sentences]
		# Count the word frequencies
		word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
		print "Found %d unique words tokens." % len(word_freq.items())
		if vocab is None:	        
			# Get the most common words and build index_to_word and word_to_index vectors
			vocab_freq = word_freq.most_common(vocabulary_size-1)
			self.vocabulary_inverted = [x[0] for x in vocab_freq]
			self.vocabulary_inverted.insert(0,UNK_IDENTIFIER)
			print "Using vocabulary size %d." % vocabulary_size
			print "The least frequent word in our vocabulary is '%s' and appeared %d times." % \
			(vocab_freq[-1][0], vocab_freq[-1][1])
		else:
			self.vocabulary_inverted = vocab
		
		
		self.vocabulary = dict([(w,i) for i,w in enumerate(self.vocabulary_inverted)])
		self.image_sentence_pairs = zip(images,tokenized_sentences)
		self.index = 0
		self.num_resets = 0
		self.num_truncates = 0
		self.num_pads = 0
		self.num_outs =0
		self.image_list=[]
		self.pad = pad
		self.truncate = truncate
		self.align = align
		if align:
			num_pairs = len(self.image_sentence_pairs)
			remainder = num_pairs % batch_num_streams
			if remainder > 0:
				num_needed = batch_num_streams - remainder
				for i in range(num_needed):
					choice = random.randint(0, num_pairs - 1)
					self.image_sentence_pairs.append(self.image_sentence_pairs[choice])
			assert len(self.image_sentence_pairs) % batch_num_streams == 0
		SequenceGenerator.__init__(self)
		self.batch_num_streams = batch_num_streams
		self.negative_one_padded_streams = frozenset(('input_sentence','target_sentence'))

	def streams_exhausted(self):
		return self.num_resets > 0
	
	def next_line(self):
		num_lines = float(len(self.image_sentence_pairs))
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
		image_filename, line = self.image_sentence_pairs[self.index]
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
		image_hash = self.image_hash(image_filename)
		out['hashed_image_path'] = [image_hash] * len(out['input_sentence'])
		self.image_list.append((image_filename, image_hash))
		self.next_line()
		return out

	def image_hash(self, filename):
		image_hash = int(sha1(filename).hexdigest(), 16) % MAX_HASH
		assert image_hash == float(image_hash)
		return image_hash


#
BUFFER_SIZE=50
OUTPUT_DIR = './models/lstm/h5_data_distill/buffer_%d' % (BUFFER_SIZE)
SPLITS_PATTERN = '/home/a-linjieyang/work/video_caption/dreamstime/%s_list.txt'
SPLITS_CAP_PATTERN = '/home/a-linjieyang/work/video_caption/dreamstime/%s_list_cap_filt.txt'
OUTPUT_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_DIR
VOCAB_OUT_PATH = './models/lstm/h5_data_distill/buffer_100/vocabulary' 
def process_dataset(split_name, dt_split_name, batch_stream_length, vocab=None):
	with open(SPLITS_PATTERN % split_name, 'r') as split_file:
		split_images = [line[:-1] for line in split_file]
	with open(SPLITS_CAP_PATTERN % split_name,'r') as split_cap_file:
		split_sentences = [line[:-1] for line in split_cap_file]
	output_dataset_name = '%s_%d' % (split_name, MAX_WORDS)
	output_path = OUTPUT_DIR_PATTERN % output_dataset_name
	sg = DtSequenceGenerator(BUFFER_SIZE, split_images, split_sentences, vocab=vocab)
	sg.batch_stream_length = batch_stream_length
	writer = HDF5SequenceWriter(sg, output_dir=output_path)
	writer.write_to_exhaustion()
	writer.write_filelists()
	if vocab is None:
		with open(VOCAB_OUT_PATH,'w') as fb:
			for w in sg.vocabulary_inverted:
				fb.write('%s\n' % w)
			#cPickle.dump(sg.vocabulary,fb)
			#cPickle.dump(sg.vocabulary_inverted,fb)
	return sg.vocabulary_inverted
def process_dreamstime():
	vocab=None
	vocab = []
	with open(VOCAB_OUT_PATH,'r') as f:
		for line in f:
			vocab.append(line.strip())
	datasets = [
	('train','train',100000),
	('val','val',100000),
	]
	for split_name, dt_split_name, batch_stream_length in datasets:
		vocab = process_dataset(split_name, dt_split_name, batch_stream_length,
			vocab = vocab)
if __name__ == "__main__":
	process_dreamstime()
