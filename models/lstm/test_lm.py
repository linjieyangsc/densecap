#!/usr/bin/env python

from collections import OrderedDict
import json
import numpy as np
import pprint
import cPickle as pickle
import string
import sys
import math

# seed the RNG so we evaluate on the same subset each time
np.random.seed(seed=0)
sys.path.append('./models/lstm/')  
from composer import Composer

def main(): 
  
  ITER = 100000
  MODEL_FILENAME = 'sent_lstm_lm2_par_iter_%d' % ITER

  DATA_DIR = 'h5_data_par/buffer_100_vocab_10000'
  MODEL_DIR = './models/lstm'
  MODEL_FILE = '%s/%s.caffemodel' % (MODEL_DIR, MODEL_FILENAME)
  LSTM_NET_FILE = './models/lstm/sent_lstm_lm2.deploy.prototxt'
  
  VOCAB_FILE = '%s/%s/vocabulary' % (MODEL_DIR, DATA_DIR)
  DEVICE_ID = 4
  with open(VOCAB_FILE, 'r') as vocab_file:
    vocab = [line.strip() for line in vocab_file.readlines()]
  
  sample_n = 1
  seed_sent = sys.argv[1]

  if (len(sys.argv)>2):
    sample_n = int(sys.argv[2])
  
  comp = Composer(MODEL_FILE, LSTM_NET_FILE, VOCAB_FILE,
                        device_id=DEVICE_ID)
  beam_size = 1
  temp = 1
  if (len(sys.argv)>3):
  	temp = int(sys.argv[3])
  #generation_strategy = {'type': 'beam', 'beam_size': sample_n}
  generation_strategy = {'type': 'sample', 'num': sample_n,'temp': temp} 
  seed_sent = comp.tokenize(seed_sent)
  samples, probs = comp.predict_caption(seed_sent, strategy=generation_strategy)
  #calculate logprob
  log_probs = np.zeros((sample_n, 1),dtype=np.float32)
  eps=1e-20
  for i in xrange(sample_n):
    for p in probs[i]:
      log_probs[i] += math.log(p+eps)
  for i in xrange(sample_n):
    print comp.sentence(samples[i])
    print 'logprob: %f' % log_probs[i]
  #experimenter.retrieval_experiment()

if __name__ == "__main__":
  main()
