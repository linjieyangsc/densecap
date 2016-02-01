import os
import cPickle
vocab_path = './h5_data/buffer_100/vocabulary'
with open(vocab_path,'rb') as f:
	vocab = cPickle.load(f)
	vocab_inv = cPickle.load(f)
sav_path = 'vocab.txt'
with open(sav_path,'w') as f:
	f.write('\n'.join(vocab_inv))

	
