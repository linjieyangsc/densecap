import os
import json
import cv2
import numpy as np
res_path = './retrieval_cache/vg_test/10_ims/dense_cap_cross3_iter_150000/sample1.0/generation_result.json'
sav_dir = 'examples/visual_genome/caption_locations1'
result = json.load(open(res_path))
for im_res in result:
  image_path = im_res['image_path']
  im_name = image_path.split('/')[-1][:-4]
  im = cv2.imread(image_path)
  for anno in im_res['caption_locations']:
    caption = anno['caption']
    print caption
    locations = np.array(anno['location_seq'])
    print locations.shape
    im_shape = im.shape
    #unnormalize to image size
    locations[:,[0,2]] *= im_shape[1]
    locations[:,[1,3]] *= im_shape[0]
    locations = locations.astype(np.int32) 
    #start displaying
    for i,location in enumerate(locations):
      im_new = np.copy(im)
      cv2.rectangle(im_new, (location[0],location[1]), (location[2],location[3]), (255,0,0), 2)
      cv2.imwrite('%s/%s_%s_%d.jpg' % (sav_dir, im_name, caption, i), im_new)
      print 'image_saved'
