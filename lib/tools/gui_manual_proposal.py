import _init_paths
import caffe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from fast_rcnn.test_cap_manual_proposal import region_captioning, sentence
from fast_rcnn.config import cfg
import cv2

class Interface(object):
    def __init__(self, image_path, image_list, vocabulary_path, feature_net, embed_net, recurrent_net, sample_n):
        self.ax = plt.axes([0,0,1,1])
        self.colors = ['g','r','c','m','y','k']
        # init rectangles to be drawn
        self.rect = Rectangle((0,0), 0, 0, fill=False, edgecolor='b')
        self.rect_predicted = []
        for i in xrange(sample_n):
            self.rect_predicted.append(Rectangle((0,0), 0, 0, fill=False, edgecolor=self.colors[i]))
            self.ax.add_patch(self.rect_predicted[-1])
        # read vocabulary
        with open(vocabulary_path,'r') as f:
            self.vocab = [line.strip() for line in f]
        self.vocab.insert(0,'<EOS>')
        # init image list and current image
        self.image_list = image_list
        self.image_id = 0
        self.image_path = image_path
        print self.image_path, self.image_list[self.image_id]
        cur_image_path = self.image_path % self.image_list[self.image_id]
        
        self.im = cv2.imread(cur_image_path)
        self.disp_im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
        
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        #self.ax.patch.set_visible(False)
        #self.ax.figure.set_visible(False)
        #self.ax.figure.set_frameon(False)
        self.ax.axis('off')
        self.ax.imshow(self.disp_im, aspect='auto')

        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.button_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.button_release)

        self.ax.figure.canvas.mpl_connect('key_press_event', self.key_press)

        
        self.feature_net = feature_net
        self.embed_net = embed_net
        self.recurrent_net = recurrent_net
        self.sample_n = sample_n
        
    def key_press(self, event):
          print '%s is pressed'% event.key
          # press s defaults to saving an image
          if event.key == 'd':
            self.image_id = (self.image_id - 1) % len(self.image_list)
          elif event.key == 'e':
            self.image_id = (self.image_id + 1) % len(self.image_list)

          if event.key in set(['d','e']):
            
            # update image to show
            cur_image_path = self.image_path % self.image_list[self.image_id]
            
            self.im = cv2.imread(cur_image_path)
            self.disp_im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
            self.ax.imshow(self.disp_im, aspect='auto')
            # reset all rectangles
            for i in range(self.sample_n):
                self.rect_predicted[i].set_width(0)
                self.rect_predicted[i].set_height(0)
                self.rect_predicted[i].set_xy((0, 0))
                self.rect_predicted[i].set_label('')
            self.rect.set_width(0)
            self.rect.set_height(0)
            self.rect.set_xy((0, 0))
            if self.ax.legend_:
                self.ax.legend_.remove()
            self.ax.figure.canvas.draw()
          #if event.key == 'w':
          #  self.im_id+=1
          #  self.ax.figure.savefig('/home/ljyang/Desktop/figure_%d.png' % self.im_id, bbox_inches='tight')
    def button_press(self, event):
        print 'press'
        self.x0 = event.xdata
        self.y0 = event.ydata

    def button_release(self, event):
        print 'release'
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        print 'rectangle coordinates: (%.01f, %.01f, %.01f, %.01f)' % (self.x0, self.y0, self.x1, self.y1)
        box = np.array([self.x0, self.y0, self.x1, self.y1]).reshape(1,4)
        #predict regressed box and caption
        strategy = {'algorithm': 'sample', 'samples': 5, 'temperature': 0.5}
        boxes, captions, _ = region_captioning(self.feature_net, self.embed_net, self.recurrent_net, self.im, box, strategy)
        #add new box
        for i, box, caption in zip(range(len(boxes)), boxes, captions):
            caption_str = sentence(self.vocab, caption)
            self.rect_predicted[i].set_width(box[2] - box[0])
            self.rect_predicted[i].set_height(box[3] - box[1])
            self.rect_predicted[i].set_xy((box[0], box[1]))
            self.rect_predicted[i].set_label(caption_str)
        for i in range(len(boxes), self.sample_n):
            self.rect_predicted[i].set_width(0)
            self.rect_predicted[i].set_height(0)
            self.rect_predicted[i].set_xy((0, 0))
            self.rect_predicted[i].set_label('')
            
        self.ax.legend(loc='best')
        self.ax.figure.canvas.draw()

def main():
    im_path = '/home/ljyang/work/data/visual_genome/images/%s.jpg'
    im_list_file = '/home/ljyang/work/data/visual_genome/densecap_splits/test.txt'
    with open(im_list_file) as f:
        im_list = [line.strip() for line in f]
    vocab_path = 'data/visual_genome/1.0/vocabulary.txt'
    caffe.set_mode_gpu()
    caffe.set_device(0)
    args={}
    args['caffemodel'] = 'output/faster_rcnn_end2end/vg_train/faster_rcnn_cap_two_stage4_512_finetune_iter_200000.caffemodel'
    
    args['feature_prototxt'] = 'models/faster_rcnn_cap/vgg_region_feature_given_box.prototxt'
    args['embed_prototxt'] = 'models/faster_rcnn_cap/test_word_embedding_512.prototxt'
    args['recurrent_prototxt'] = 'models/faster_rcnn_cap/test_cap_pred4_512.prototxt'
    feature_net = caffe.Net(args['feature_prototxt'], args['caffemodel'], caffe.TEST)
    embed_net = caffe.Net(args['embed_prototxt'], args['caffemodel'], caffe.TEST)
    recurrent_net = caffe.Net(args['recurrent_prototxt'], args['caffemodel'], caffe.TEST)
    sample_n = 5
    cfg.TEST.HAS_RPN=False
    cfg.MAX_SIZE = 720

    a = Interface(im_path, im_list, vocab_path, feature_net, embed_net, recurrent_net, sample_n)
    plt.show()

if __name__ =='__main__':
    main()