#!/usr/bin/env python

from collections import OrderedDict
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys

sys.path.append('./python/')
import caffe
REMOVE_UNK = True
class Composer():
  def __init__(self, weights_path, lstm_net_proto,
               vocab_path, device_id=-1):
    if device_id >= 0:
      caffe.set_mode_gpu()
      caffe.set_device(device_id)
    else:
      caffe.set_mode_cpu()
    # Setup image processing net.
    phase = caffe.TEST
    #self.image_net = caffe.Net(image_net_proto, weights_path, phase)
    #image_data_shape = self.image_net.blobs['data'].data.shape
    #self.transformer = caffe.io.Transformer({'data': image_data_shape})
    #channel_mean = np.zeros(image_data_shape[1:])
    #channel_mean_values = [104, 117, 123]
    #assert channel_mean.shape[0] == len(channel_mean_values)
    #for channel_index, mean_val in enumerate(channel_mean_values):
    #  channel_mean[channel_index, ...] = mean_val
    #self.transformer.set_mean('data', channel_mean)
    #self.transformer.set_channel_swap('data', (2, 1, 0))
    #self.transformer.set_transpose('data', (2, 0, 1))
    # Setup sentence prediction net.
    self.lstm_net = caffe.Net(lstm_net_proto, weights_path, phase)
    self.vocab = ['<EOS>']
    with open(vocab_path, 'r') as vocab_file:
      self.vocab += [word.strip() for word in vocab_file.readlines()]
    assert(self.vocab[1] == '<unk>')
    net_vocab_size = self.lstm_net.blobs['predict'].data.shape[2]
    if len(self.vocab) != net_vocab_size:
      raise Exception('Invalid vocab file: contains %d words; '
          'net expects vocab with %d words' % (len(self.vocab), net_vocab_size))
    self.vocab_inv = dict([(w,i) for i,w in enumerate(self.vocab)])
  
  def caption_batch_size(self):
    return self.lstm_net.blobs['cont_sentence'].data.shape[1]

  def set_caption_batch_size(self, batch_size):
    self.lstm_net.blobs['cont_sentence'].reshape(1, batch_size)
    self.lstm_net.blobs['input_sentence'].reshape(1, batch_size)
    #self.lstm_net.blobs['image_features'].reshape(batch_size,
     #   *self.lstm_net.blobs['image_features'].data.shape[1:])
    self.lstm_net.reshape()


  def predict_single_word(self, previous_word, output='probs'):
    net = self.lstm_net
    cont = 0 if previous_word == 0 else 1
    cont_input = np.array([cont])
    word_input = np.array([previous_word])
    #image_features = np.zeros_like(net.blobs['image_features'].data)
    #image_features[:] = descriptor
    net.forward(cont_sentence=cont_input,
                input_sentence=word_input)
    output_preds = net.blobs[output].data[0, 0, :]

    return output_preds


    return output_preds
  def predict_single_word_from_all_previous(self, previous_words):
    for word in [0] + previous_words:
      probs = self.predict_single_word(word)
    return probs

  # Strategy must be either 'beam' or 'sample'.
  # If 'beam', do a max likelihood beam search with beam size num_samples.
  # Otherwise, sample with temperature temp.
  def predict_caption(self, seed_sent, strategy={'type': 'beam'}):
    assert 'type' in strategy
    assert strategy['type'] in ('beam', 'sample')
    if strategy['type'] == 'beam':
      return self.predict_caption_beam_search(seed_sent, strategy)
    num_samples = strategy['num'] if 'num' in strategy else 1
    samples = []
    sample_probs = []
    for _ in range(num_samples):
      sample, sample_prob = self.sample_caption(seed_sent, strategy)
      samples.append(sample)
      sample_probs.append(sample_prob)
    return samples, sample_probs

  def sample_caption(self, seed_sent, strategy,
                     net_output='predict', max_length=100):
    sentence = []
    probs = []
    eps_prob = 1e-8
    temp = strategy['temp'] if 'temp' in strategy else 1.0
    #if max_length < 0: max_length = float('inf')
    start=True
    while len(sentence) < max_length:
      
      if start:
        softmax_inputs = self.predict_single_word_from_all_previous(seed_sent)
        start=False
      else:
        previous_word = sentence[-1] if sentence else 0
        softmax_inputs = self.predict_single_word(previous_word,
                                                output=net_output)
      word = random_choice_from_probs(softmax_inputs, temp)
      sentence.append(word)
      probs.append(softmax(softmax_inputs, 1.0)[word])
    return sentence, probs

  def predict_caption_beam_search(self, seed_sent, strategy, max_length=100):
    orig_batch_size = self.caption_batch_size()
    if orig_batch_size != 1: self.set_caption_batch_size(1)
    beam_size = strategy['beam_size'] if 'beam_size' in strategy else 1
    assert beam_size >= 1
    beams = [[]]
    beams_complete = 0
    beam_probs = [[]]
    beam_log_probs = [0.]
    start=True
    while beams_complete < len(beams):
      expansions = []
      for beam_index, beam_log_prob, beam in \
          zip(range(len(beams)), beam_log_probs, beams):
        if beam:
          previous_word = beam[-1]
          if len(beam) >= max_length:
            exp = {'prefix_beam_index': beam_index, 'extension': [],
                   'prob_extension': [], 'log_prob': beam_log_prob}
            expansions.append(exp)
            # Don't expand this beam; it was already ended with an EOS,
            # or is the max length.
            continue

          if beam_size == 1:
            probs = self.predict_single_word(previous_word)
          else:
            probs = self.predict_single_word_from_all_previous(seed_sent+beam)
        else:
          
          probs = self.predict_single_word_from_all_previous(seed_sent)
          
        assert len(probs.shape) == 1
        assert probs.shape[0] == len(self.vocab)
        expansion_inds = probs.argsort()[-beam_size:]
        for ind in expansion_inds:
          if not (REMOVE_UNK and ind==1):
            assert(self.vocab[ind] !='<unk>')
            prob = probs[ind]
            extended_beam_log_prob = beam_log_prob + math.log(prob)
            exp = {'prefix_beam_index': beam_index, 'extension': [ind],
         'prob_extension': [prob], 'log_prob': extended_beam_log_prob}
            expansions.append(exp)
      # Sort expansions in decreasing order of probability.
      expansions.sort(key=lambda expansion: -1 * expansion['log_prob'])
      expansions = expansions[:beam_size]
      new_beams = \
          [beams[e['prefix_beam_index']] + e['extension'] for e in expansions]
      new_beam_probs = \
          [beam_probs[e['prefix_beam_index']] + e['prob_extension'] for e in expansions]
      beam_log_probs = [e['log_prob'] for e in expansions]
      beams_complete = 0
      for beam in new_beams:
        if len(beam) >= max_length: beams_complete += 1
      beams, beam_probs = new_beams, new_beam_probs
    if orig_batch_size != 1: self.set_caption_batch_size(orig_batch_size)
    return beams, beam_probs

  def sentence(self, vocab_indices):
    sentence = ' '.join([self.vocab[i] for i in vocab_indices])
    if not sentence: return sentence
    #sentence = sentence[0].upper() + sentence[1:]
    # If sentence ends with ' <EOS>', remove and replace with '.'
    # Otherwise (doesn't end with '<EOS>' -- maybe was the max length?):
    # append '...'
    suffix = ' ' + self.vocab[0]
    if sentence.endswith(suffix):
      sentence = sentence[:-len(suffix)] + '.'
    else:
      sentence += '...'
    return sentence
  
  def tokenize(self, sentence):
    tokens = sentence.split()
    sent_token = []
    for token in tokens:
      if token in self.vocab_inv:
        sent_token.append(self.vocab_inv[token])
      else:
        sent_token.append(1) #unknown tag
    return sent_token

def softmax(softmax_inputs, temp):
  shifted_inputs = softmax_inputs - softmax_inputs.max()
  exp_outputs = np.exp(temp * shifted_inputs)
  exp_outputs_sum = exp_outputs.sum()
  if math.isnan(exp_outputs_sum):
    return exp_outputs * float('nan')
  assert exp_outputs_sum > 0
  if math.isinf(exp_outputs_sum):
    return np.zeros_like(exp_outputs)
  eps_sum = 1e-20
  return exp_outputs / max(exp_outputs_sum, eps_sum)

def random_choice_from_probs(softmax_inputs, temp=1, already_softmaxed=False):
  # temperature of infinity == take the max
  if REMOVE_UNK:
    if already_softmaxed:
      softmax_inputs[1] = 0.0
      softmax_inputs = softmax_inputs / softmax_inputs.sum()
    else:
      softmax_inputs[1] = -20.0

  if temp == float('inf'):
    return np.argmax(softmax_inputs)
  if already_softmaxed:
    probs = softmax_inputs
    assert temp == 1
  else:
    probs = softmax(softmax_inputs, temp)
  r = random.random()
  
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 0  # return eos?

def gen_stats(prob, normalizer=None):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += math.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  stats['p'] = math.exp(stats['log_p'])
  stats['p_word'] = math.exp(stats['log_p'])
  try:
    stats['perplex'] = math.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = math.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  if normalizer is not None:
    norm_stats = gen_stats(normalizer)
    stats['normed_perplex'] = stats['perplex'] / norm_stats['perplex']
    stats['normed_perplex_word'] = \
        stats['perplex_word'] / norm_stats['perplex_word']
  return stats
