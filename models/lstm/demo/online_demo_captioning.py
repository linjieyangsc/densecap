#!/usr/bin/env python
import json
import numpy as np
import time
import sys
from captioner import Captioner
# import matplotlib.pyplot as plt
from PIL import Image

class CaptioningService():
  def __init__(self):
    #initialize networks
    ITER = 100000
    MODEL_FILENAME = 'lrcn_finetune_googlenet_iter_%d' % ITER

    MODEL_FILE = '%s.caffemodel' % (MODEL_FILENAME)
    IMAGE_NET_FILE = 'googlenet.deploy.prototxt'
    WORD_EMBEDDING_FILE = 'word_embedding.prototxt'
    LSTM_NET_FILE = 'lrcn_word_to_preds.min.deploy.prototxt'
    VOCAB_FILE = 'vocabulary'
    # DEVICE_ID < 0 for CPU, otherwise for GPU
    DEVICE_ID = -1
    
    self.captioner = Captioner(MODEL_FILE, IMAGE_NET_FILE, WORD_EMBEDDING_FILE, LSTM_NET_FILE, VOCAB_FILE,
                        device_id=DEVICE_ID)
  def process(self, image):
    #record the processing time
    t1 = time.time()
    descriptor = self.captioner.compute_descriptor(image, output_name='image_feature')
    t2 = time.time()
    caption, log_prob = self.captioner.predict_caption(descriptor)
    res = self.format_json(caption, log_prob)
    t3 = time.time()
    print 'image feature processing time: %.03f seconds' % (t2-t1)
    print 'caption generation time: %.03f seconds' % (t3-t2)
    return res
  def format_json(self, caption, log_prob):
    res = {}
    res['results'] = {'caption_v1': {'confs':[log_prob], 'tags':[caption]}}
    return json.dumps(res)

def main(image_path):
  #initilize
  caption_service = CaptioningService()
  #process an input

  # image = plt.imread(image_path)
  image = Image.open(image_path)
  image = np.array(image)
  print caption_service.process(image)

if __name__ == "__main__":
  #use path of an image to test  
  if len(sys.argv) < 2:
    raise Exception('Paramter error. Usage: python online_demo_captioning.py path_to_an_image')
  main(sys.argv[1])
