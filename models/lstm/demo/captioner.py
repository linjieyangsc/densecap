#!/usr/bin/env python
import math

import numpy as np
# need a compiled caffe libruary
import caffe
REMOVE_UNK = True
class Captioner():
  def __init__(self, weights_path, image_net_proto, word_embed_net_proto, lstm_net_proto,
               vocab_path, device_id=-1):
    if device_id >= 0:
      caffe.set_mode_gpu()
      caffe.set_device(device_id)
    else:
      caffe.set_mode_cpu()
    # Setup image processing net.
    phase = caffe.TEST
    self.image_net = caffe.Net(image_net_proto, weights_path, phase)
    image_data_shape = self.image_net.blobs['data'].data.shape
    self.transformer = caffe.io.Transformer({'data': image_data_shape})
    channel_mean = np.zeros(image_data_shape[1:])
    channel_mean_values = [104, 117, 123]
    assert channel_mean.shape[0] == len(channel_mean_values)
    for channel_index, mean_val in enumerate(channel_mean_values):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data', channel_mean)
    self.transformer.set_channel_swap('data', (2, 1, 0))
    self.transformer.set_transpose('data', (2, 0, 1))
    # Setup sentence prediction net.
    self.word_embed_net = caffe.Net(word_embed_net_proto, weights_path, phase)
    self.lstm_net = caffe.Net(lstm_net_proto, weights_path, phase)
    self.vocab = ['<EOS>']
    with open(vocab_path, 'r') as vocab_file:
      self.vocab += [word.strip() for word in vocab_file.readlines()]
    assert(self.vocab[1] == '<unk>')
    self.vocab_inv = dict([(w,i) for i,w in enumerate(self.vocab)])
    net_vocab_size = self.lstm_net.blobs['predict'].data.shape[2]
    if len(self.vocab) != net_vocab_size:
      raise Exception('Invalid vocab file: contains %d words; '
          'net expects vocab with %d words' % (len(self.vocab), net_vocab_size))


  def preprocess_image(self, image):

    # crop_edge_ratio = (256. - 227.) / 256. / 2
    # ch = int(image.shape[0] * crop_edge_ratio + 0.5)
    # cw = int(image.shape[1] * crop_edge_ratio + 0.5)
    # cropped_image = image[ch:-ch, cw:-cw]

    #no cropping
    cropped_image = image
    if len(cropped_image.shape) == 2:
      cropped_image = np.tile(cropped_image[:, :, np.newaxis], (1, 1, 3))
    preprocessed_image = self.transformer.preprocess('data', cropped_image)
   
    return preprocessed_image

  def preprocessed_image_to_descriptor(self, image, output_name='fc8'):
    net = self.image_net
    if net.blobs['data'].data.shape[0] > 1:
      batch = np.zeros_like(net.blobs['data'].data)
      batch[0] = image[0]
    else:
      batch = image
    net.forward(data=batch)
    descriptor = net.blobs[output_name].data[0].copy()
    return descriptor

  def predict_single_word(self, previous_word, output='probs'):
    net = self.lstm_net
    cont = 1
    cont_input = np.array([cont])
    word_input = np.array([previous_word])
    embed_out = self.word_embed_net.forward(input_sentence=word_input)
    
    #input_features = np.zeros_like(net.blobs['input_features'].data)
    input_features = embed_out['word_embedding']
    net.forward(input_features=input_features, cont_sentence=cont_input)
    output_preds = net.blobs[output].data[0, 0, :]
 
    return output_preds



  # predict caption with beam-1 search
  def predict_caption(self, descriptor, max_length=30):

    caption = []
    eps=1e-12
    caption_log_prob = 0.0
    # first time step
    net = self.lstm_net
    input_features = np.zeros_like(net.blobs['input_features'].data)
    input_features[:] = descriptor
    cont_input = np.array([0])
    net.forward(input_features=input_features, cont_sentence=cont_input)
    while (not caption or caption[-1] != 0) and len(caption) < max_length:
      if not caption:
        previous_word = 0
      else:
        previous_word = caption[-1]
      probs = self.predict_single_word(previous_word)
      if REMOVE_UNK:
        probs[1] = 0.0
      next_id = probs.argmax()
      prob = probs[next_id]
      caption.append(next_id)
      caption_log_prob += math.log(max(prob,eps))
    caption_str = self.sentence(caption)
    return caption_str, caption_log_prob


  

  def compute_descriptor(self, image, output_name='fc8'):

      

    preprocessed_image = self.preprocess_image(image)
    image_input = np.zeros_like(self.image_net.blobs['data'].data)
    image_input[...] = preprocessed_image
    self.image_net.forward(data=image_input)
    descriptor = self.image_net.blobs[output_name].data[0].copy()
    return descriptor



  def sentence(self, vocab_indices):
    sentence = ' '.join([self.vocab[i] for i in vocab_indices])
    if not sentence: return sentence
    sentence = sentence[0].upper() + sentence[1:]
    # If sentence ends with ' <EOS>', remove and replace with '.'
    # Otherwise (doesn't end with '<EOS>' -- maybe was the max length?):
    # append '...'
    suffix = ' ' + self.vocab[0]
    if sentence.endswith(suffix):
      sentence = sentence[:-len(suffix)] + '.'
    else:
      sentence += '...'
    return sentence
  



