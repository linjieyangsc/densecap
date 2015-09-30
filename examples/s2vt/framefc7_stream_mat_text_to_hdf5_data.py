#!/usr/bin/env python
# Use this script to process data if the features are in a matfile.

import csv
import numpy as np
import os
import random
random.seed(3)
import sys
import getopt
import shutil 
import h5py

from hdf5_npstreamsequence_generator import SequenceGenerator, HDF5SequenceWriter

# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = '<en_unk>'

# start every sentence in a new array, pad if <max
FEAT_DIM= 4096

"""Filenames has file with vgg fc7 frame feats for vidids
   and sentences with video ids"""
class fc7FrameSequenceGenerator(SequenceGenerator):
  def __init__(self, feat_mat, filenames, buffer_size,  batch_num_streams=1, vocab_filename=None,
               max_words=80, align=True, shuffle=True, pad=True,
               truncate=True, reverse=False, mean_pool_frames=False, randomize_frames=False):
    self.max_words = max_words
    self.reverse = reverse
    self.array_type_inputs = {} # stream inputs that are arrays
    self.array_type_inputs['frame_fc7'] = FEAT_DIM # stream inputs that are arrays
    self.lines = []
    self.mean_pool_frames = mean_pool_frames
    self.randomize_frames = randomize_frames
    num_empty_lines = 0
    self.vid_framefeats = {} # listofdict [{}]
    
    for framefeatfile, sentfile in filenames:
      print 'Reading frame features from file: %s' % framefeatfile
      total_counter = 0
      with open(framefeatfile, 'rb') as featfd:
        # each line has the fc7 for 1 frame in video
        for line in featfd:
          line = line.strip()
          dot_folder_folder_filename = line.split('/')
          video_id = '{0}/{1}'.format(dot_folder_folder_filename[-3],dot_folder_folder_filename[-2])
          assert feat_mat.shape[1] == FEAT_DIM , 'featmat is of wrong shape: {0}, expected {1}'.format(feat_mat.shape[1],FEAT_DIM)
          if video_id not in self.vid_framefeats:
            if self.mean_pool_frames:
              self.vid_framefeats[video_id]=0
            else:
              self.vid_framefeats[video_id]=[]
          if self.mean_pool_frames:
            # first just count the number of frames
            self.vid_framefeats[video_id] += 1
          else:
            self.vid_framefeats[video_id].append(feat_mat[total_counter,:].reshape(1, FEAT_DIM))
          total_counter+=1
          
      if self.mean_pool_frames:
        total_counter = 0
        for video_id, n_frames in self.vid_framefeats.iteritems():
          #import ipdb; ipdb.set_trace(); 
          self.vid_framefeats[video_id] = np.mean(feat_mat[total_counter:total_counter+n_frames-1], 0).reshape(1, FEAT_DIM)
          total_counter += n_frames
        

      # reset max_words based on maximum frames in the video
      print 'Reading sentences in: %s' % sentfile
      with open(sentfile, 'r') as sentfd:
        for line in sentfd:
          line = line.strip()
          id_sent = line.split('\t')
          if len(id_sent)<2:
            num_empty_lines += 1
            continue
          self.lines.append((id_sent[0], id_sent[1]))
      if num_empty_lines > 0:
        print 'Warning: ignoring %d empty lines.' % num_empty_lines
    self.line_index = 0
    self.num_resets = 0
    self.num_truncates = 0
    self.num_pads = 0
    self.num_outs = 0
    self.frame_list = []
    self.vocabulary = {}
    self.vocabulary_inverted = []
    self.vocab_counts = []
    # initialize vocabulary
    self.init_vocabulary(vocab_filename)
    SequenceGenerator.__init__(self)
    self.batch_num_streams = batch_num_streams  # needed in hdf5 to seq
    # make the number of image/sentence pairs a multiple of the buffer size
    # so each timestep of each batch is useful and we can align the images
    if align:
      num_pairs = len(self.lines)
      remainder = num_pairs % buffer_size
      if remainder > 0:
        num_needed = buffer_size - remainder
        for i in range(num_needed):
          choice = random.randint(0, num_pairs - 1)
          self.lines.append(self.lines[choice])
      assert len(self.lines) % buffer_size == 0
    if shuffle:
      random.shuffle(self.lines)
    self.pad = pad
    self.truncate = truncate
    self.negative_one_padded_streams = frozenset(('target_sentence'))

  def streams_exhausted(self):
    return self.num_resets > 0

  def init_vocabulary(self, vocab_filename):
    print "Initializing the vocabulary from {0}.".format(vocab_filename)
    if os.path.isfile(vocab_filename):
      with open(vocab_filename, 'rb') as vocab_file:
        self.init_vocab_from_file(vocab_file)
    else:
      self.init_vocabulary_from_data(vocab_filename)

  def init_vocab_from_file(self, vocab_filedes):
    # initialize the vocabulary with the UNK word
    self.vocabulary = {UNK_IDENTIFIER: 0}
    self.vocabulary_inverted = [UNK_IDENTIFIER]
    num_words_dataset = 0
    for line in vocab_filedes.readlines():
      split_line = line.split()
      word = split_line[0]
      print word
      if word == UNK_IDENTIFIER:
        continue
      else:
        assert word not in self.vocabulary
      num_words_dataset += 1
      self.vocabulary[word] = len(self.vocabulary_inverted)
      self.vocabulary_inverted.append(word)
    num_words_vocab = len(self.vocabulary.keys())
    print ('Initialized vocabulary from file with %d unique words ' +
           '(from %d total words in dataset).') % \
          (num_words_vocab, num_words_dataset)
    assert len(self.vocabulary_inverted) == num_words_vocab

  def init_vocabulary_from_data(self, vocab_filename):
    print 'Initializing the vocabulary from full data'
    assert len(self.lines) > 0
    # initialize the vocabulary with the UNK word if new
    self.vocabulary = {UNK_IDENTIFIER: 0}
    self.vocabulary_inverted.append(UNK_IDENTIFIER)
    # count frequency of word in data
    self.vocab_counts.append(0)
      
    num_words_dataset = 0
    for vidid, line in self.lines:
      split_line = line.split()
      num_words_dataset += len(split_line)
      for word in split_line:
        if word in self.vocabulary:
          self.vocab_counts[self.vocabulary[word]] += 1
        else:
          self.vocabulary_inverted.append(word)
          self.vocabulary[word] = len(self.vocab_counts)
          self.vocab_counts.append(1)
          
    num_words_vocab = len(self.vocabulary.keys())
    print ('Initialized the vocabulary from data with %d unique words ' +
           '(from %d total words in dataset).') % (num_words_vocab, num_words_dataset)
    assert len(self.vocab_counts) == num_words_vocab
    assert len(self.vocabulary_inverted) == num_words_vocab
    if self.vocab_counts[self.vocabulary[UNK_IDENTIFIER]] == 0:
      print 'Warning: the count for the UNK identifier "%s" was 0.' % UNK_IDENTIFIER

  def dump_vocabulary(self, vocab_filename):
    print 'Dumping vocabulary to file: %s' % vocab_filename
    with open(vocab_filename, 'wb') as vocab_file:
      for word in self.vocabulary_inverted:
        vocab_file.write('%s\n' % word)
    print 'Done.'

  def dump_video_file(self, vidid_order_file, frame_seq_label_file):
    print 'Dumping vidid order to file: %s' % vidid_order_file
    with open(vidid_order_file,'wb') as vidid_file:
      for vidid, line in self.lines:
        word_count = len(line.split())
        frame_count = len(self.vid_framefeats[vidid])
        total_count = word_count + frame_count
        vidid_file.write('%s\t%d\t%d\t%d\n' % (vidid, word_count, frame_count, total_count))
    print 'Done.' 

  def next_line(self):
    num_lines = float(len(self.lines))
    if self.line_index == 1 or self.line_index == num_lines or self.line_index % 10000 == 0:
      print 'Processed %d/%d (%f%%) lines' % (self.line_index, num_lines,
                                              100 * self.line_index / num_lines)
    self.line_index += 1
    if self.line_index == num_lines:
      self.line_index = 0
      self.num_resets += 1

  def get_pad_value(self, stream_name):
    return -1 if stream_name in self.negative_one_padded_streams else 0


  def float_line_to_stream(self, line):
    return map(float, line.split(','))

  def line_to_stream(self, sentence):
    stream = []
    for word in sentence.split():
      word = word.strip()
      if word in self.vocabulary:
        stream.append(self.vocabulary[word])
      else:  # unknown word; append UNK
        stream.append(self.vocabulary[UNK_IDENTIFIER])
    # increment the stream -- 0 will be the EOS character
    stream = [s + 1 for s in stream]
    return stream

  def get_mean_frames_streams(self):
    vidid, line = self.lines[self.line_index]
    assert vidid in self.vid_framefeats
    feats_vgg_fc7 = self.vid_framefeats[vidid] # mean pooled fc7 feats for all frames
    
    stream = self.line_to_stream(line)
    num_words = len(stream)
    pad = self.max_words - (num_words + 1 ) if self.pad else 0
    truncated = False
    if pad < 0:
      # print '{0} #frames: {1} #words: {2}'.format(vidid, num_frames, num_words)
      # truncate frames
      if num_words > 20 and (num_words + 1) >= self.max_words:
        stream = stream[:20] # truncate words to 20
      truncated = True
      pad = 0
      # print 'Truncated_{0}: #frames: {1} #words: {2}'.format(vidid, num_frames, len(stream))
      self.num_truncates += truncated

    # reverse the string
    if self.reverse:
      stream.reverse()

    if pad > 0: self.num_pads += 1
    self.num_outs += 1
    out = {}
    out['cont_sentence'] =    [0] + [1] * ( len(stream)) + [0] * pad
    out['input_sentence'] =   [0] + stream + [0] * pad
    out['target_sentence'] =  stream + [0] + [-1] * pad
    # For encoder-decoder model
    out['stage_indicator'] =  [1] * (len(stream) + 1 + pad)
    out['inv_stage_indicator']=[0]* (len(stream) + 1 + pad)
    # fc7 features
    out['frame_fc7'] = []

    frame_index_sequence = range(len(stream))
    for frame_feat_indx in frame_index_sequence:
      out['frame_fc7'].append(feats_vgg_fc7)
    # pad last frame for the length of the sentence
    num_img_pads = len(out['input_sentence']) - len(stream)
    zero_padding = np.zeros_like(feats_vgg_fc7)
    for padframe in range(num_img_pads):
      out['frame_fc7'].append(zero_padding)
    assert len(out['frame_fc7'])==len(out['input_sentence'])
    return out
      
  def get_streams(self):
    if self.mean_pool_frames:
      out = self.get_mean_frames_streams()
    else:
      vidid, line = self.lines[self.line_index]
      assert vidid in self.vid_framefeats
      feats_vgg_fc7 = self.vid_framefeats[vidid] # list of fc7 feats for all frames
      num_frames = len(feats_vgg_fc7)
      stream = self.line_to_stream(line)
      num_words = len(stream)
      pad = self.max_words - (num_words + 1 + num_frames) if self.pad else 0
      truncated = False
      if pad < 0:
        # print '{0} #frames: {1} #words: {2}'.format(vidid, num_frames, num_words)
        # truncate frames
        if num_words > 20 and (num_words + num_frames + 1) >= self.max_words:
          stream = stream[:20] # truncate words to 20
        num_frames = self.max_words - (len(stream)+1)
        truncated = True
        pad = 0
        # print 'Truncated_{0}: #frames: {1} #words: {2}'.format(vidid, num_frames, len(stream))
        self.num_truncates += truncated
  
      # reverse the string
      if self.reverse:
        stream.reverse()
  
      if pad > 0: self.num_pads += 1
      self.num_outs += 1
      out = {}
      out['cont_sentence'] = [0] + [1] * (num_frames +len(stream)) + [0] * pad
      out['input_sentence'] = [0] * num_frames + [0] + stream + [0] * pad
      out['target_sentence'] = [-1] * num_frames + stream + [0] + [-1] * pad
      # For encoder-decoder model
      out['cont_img'] = [0] + [1] * (num_frames - 1) + [0] * (len(stream) + 1 + pad)
      out['cont_sen'] = [0] * (num_frames + 1) + [1] * len(stream) + [0] * pad
      out['encoder_to_decoder'] = [0] * (num_frames - 1) + [1] + [0] * (len(stream) + 1 + pad)
      out['stage_indicator'] = [0] * num_frames + [1] * (len(stream) + 1 + pad)
      out['inv_stage_indicator'] = [1] * num_frames + [0] * (len(stream) + 1 + pad)
      # fc7 features
      out['frame_fc7'] = []
      if self.randomize_frames:
        frame_index_sequence = list(np.random.permutation(num_frames))
      else:
        frame_index_sequence = range(num_frames)
      for frame_feat_indx in frame_index_sequence:
        out['frame_fc7'].append(feats_vgg_fc7[frame_feat_indx])
      # pad last frame for the length of the sentence
      num_img_pads = len(out['input_sentence']) - num_frames
      zero_padding = np.zeros_like(feats_vgg_fc7[0])
      for padframe in range(num_img_pads):
        out['frame_fc7'].append(zero_padding)
      assert len(out['frame_fc7'])==len(out['input_sentence'])
    self.next_line()
    return out

def generate_hd5(scriptname, argv):
  # start every sentence in a new array, pad if <20
  max_words = 80 
  buffer_size = 16
  batch_stream_length = 1000
  setting = 'mvad'  # Either md (for mpii-md) or mvad.
  data_setting = 'frames05'
  frame_setting = 'nr'
  randomize_frames = False
  mean_pool_frames = False
  
  # NOTE: Copy data to by creating the following directory structure.
  framefeat_file_pattern = './{0}/jpgAllFrames/frameslist_05.txt'
  sentences_file_pattern = './{0}/rawcorpus/{1}OneFile.tsv'
  mat_filename = './{0}/rawcorpus/{1}_vgg_filelist.mat'
  
  usage = '%s -b <buffer_size> -s <setting> -d <data_setting> -l <batch_stream_length> -f <framefeat_file_pattern> -w <max_words> -m <mat_filename> -r[randomize_frames] -p[mean_pool]' % scriptname
  default_values = '%s -b %d -s %s -d %s -l %d -f %s -w %d -m %s  -r -p'  % (scriptname,buffer_size,setting,data_setting,batch_stream_length,framefeat_file_pattern,max_words,mat_filename)
  
  
  
  try:
    opts, args = getopt.getopt(argv,'hb:s:l:d:f:w:m:rp')
  except getopt.GetoptError:
    print "Usage:"
    print usage
    print "Default values:"
    print default_values
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print "Usage:"
      print usage
      print "Default values:"
      print default_values
      sys.exit()
    elif opt in ('-b'):
      buffer_size = int(arg)
    elif opt in ('-s'):
      setting = arg
    elif opt in ('-l'):
      batch_stream_length = int(arg)
    elif opt in ('-d'):
      data_setting = arg
    elif opt in ('-f'):
      framefeat_file_pattern = arg
    elif opt in ('-w'):
      max_words = int(arg)
    elif opt in ('-m'):
      mat_filename = arg 
    elif opt in ('-r'):
      randomize_frames = True
      frame_setting = 'rand'
    elif opt in ('-p'):
      mean_pool_frames = True
      frame_setting = 'mean'
      max_words = 21
      
      
  
  OUTHOME = '.'
  output_dir = '{0}/{1}/generated_data/prevgg_b{2}_w{3}_{4}_{5}'.format(
      OUTHOME, setting, buffer_size, max_words, data_setting, frame_setting)
  vocab_filename = '{0}/vocabulary.txt'.format(output_dir)
  outdir_pattern = '{0}/%s/'.format(output_dir)
  h5_dir_pattern = '{0}/batches/'.format(outdir_pattern)
  
  DATASETS = [ # split_name, data_split_name, aligned
      ('train', 'training', False, False),
      ('valid', 'val', False, False),
  ]
  f = h5py.File(mat_filename.format(setting,data_setting),'r')
  featmat = np.array(f["scores"])
  for split_name, data_split_name, aligned, reverse in DATASETS:
    filenames = [(framefeat_file_pattern.format(setting,data_setting),
               sentences_file_pattern.format(setting,data_split_name))]
    
    output_path = h5_dir_pattern % split_name
    fsg = fc7FrameSequenceGenerator(featmat,filenames, buffer_size, batch_num_streams=buffer_size,
           vocab_filename=vocab_filename,
           max_words=max_words, align=aligned, shuffle=True, pad=aligned,
           truncate=aligned, reverse=reverse, randomize_frames=randomize_frames, mean_pool_frames=mean_pool_frames)
    fsg.batch_stream_length = batch_stream_length
    if os.path.exists(output_path):
      shutil.rmtree(output_path)
    writer = HDF5SequenceWriter(fsg, output_dir=output_path)
    writer.write_to_exhaustion()
    writer.write_filelists()
    if not os.path.isfile(vocab_filename):
      print "Vocabulary not found"
      # fsg.dump_vocabulary(vocab_out_path)
    if not os.path.isfile(vocab_filename):
      fsg.dump_vocabulary(vocab_filename)
    out_path = outdir_pattern % split_name
    vid_id_order_outpath = '%s/vidid_order.txt' % out_path
    frame_sequence_outpath = '%s/frame_sequence.txt' % out_path
    fsg.dump_video_file(vid_id_order_outpath, frame_sequence_outpath)

if __name__ == "__main__":
  np.random.seed(42)
  generate_hd5(sys.argv[0], sys.argv[1:])
