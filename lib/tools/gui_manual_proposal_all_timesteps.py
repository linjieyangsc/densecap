import _init_paths
import caffe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from fast_rcnn.test_cap_manual_proposal import region_captioning, sentence
from fast_rcnn.config import cfg
import cv2

class Interface(object):
    def __init__(self, image_path, image_list, vocabulary_path, feature_net, embed_net, recurrent_net, max_timesteps):
        self.ax = plt.axes([0,0,1,1])
        #colors should be a linear transformation of one to another
        self.start_color = np.array((1, 0.5, 0.)) # blue
        self.end_color = np.array((1, 1, 0.5)) # magnita
        #self.colors = ['g','r','c','m','y','k']
        # init rectangles to be drawn
        lw=4
        self.rect = Rectangle((0,0), 0, 0, fill=False, edgecolor=self.start_color, linewidth=lw)
        self.rect.set_label('proposal')
        self.ax.add_patch(self.rect)
        self.rect_predicted = []
        for i in xrange(max_timesteps):
            self.rect_predicted.append(Rectangle((0,0), 0, 0, fill=False, linewidth=lw))
            self.ax.add_patch(self.rect_predicted[-1])
        # read vocabulary
        with open(vocabulary_path,'r') as f:
            self.vocab = [line.strip() for line in f]
        self.vocab.insert(0,'<EOS>')
        # init image list and current image
        self.image_list = image_list
        self.image_id = 0
        self.image_path = image_path
        self.legend_loc= {'1':'upper left', '2':'upper right','3':'lower left','4':'lower right'}
        print self.image_path, self.image_list[self.image_id]
        print 'loading image #%d, name: %s' % (self.image_id, self.image_list[self.image_id])
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

        
        self.ax.figure.canvas.mpl_connect('button_press_event', self.button_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.button_release)

        self.ax.figure.canvas.mpl_connect('key_press_event', self.key_press)

        
        self.feature_net = feature_net
        self.embed_net = embed_net
        self.recurrent_net = recurrent_net
        self.max_timesteps = max_timesteps
        
    def key_press(self, event):
          #print '%s is pressed'% event.key
          # press s defaults to saving an image
          if event.key == 'd':
            self.image_id = (self.image_id - 1) % len(self.image_list)
          elif event.key == 'e':
            self.image_id = (self.image_id + 1) % len(self.image_list)
          elif event.key == 'g':
            self.image_id = (self.image_id - 10) % len(self.image_list)
          elif event.key == 't':
            self.image_id = (self.image_id + 10) % len(self.image_list)
          elif event.key == 'q':
            self.image_name = raw_input()
          if event.key in set(['d','e','g','t', 'q']):
            if event.key == 'q':
                cur_image_path = self.image_path % self.image_name
            else:
          
                # update image to show
                print 'loading image #%d, name: %s' % (self.image_id, self.image_list[self.image_id])
                cur_image_path = self.image_path % self.image_list[self.image_id]
            
            self.im = cv2.imread(cur_image_path)
            self.disp_im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
            self.ax.imshow(self.disp_im, aspect='auto')
            # reset all rectangles
            for i in range(self.max_timesteps):
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
          elif event.key in set(['1','2','3','4']):
            self.ax.legend(loc=self.legend_loc[event.key],prop={'size':22})
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
        strategy = {'algorithm': 'greedy'}
        _, captions, boxes = region_captioning(self.feature_net, self.embed_net, self.recurrent_net, self.im, box, strategy)
        #add new box
        #caption_str = sentence(self.vocab, captions[0])
        assert(len(captions[0]) == boxes.shape[0])
        words = [self.vocab[w] for w in captions[0][:-1]]
        words.insert(0,'<SOS>')
        for i, box, word in zip(range(len(boxes)), boxes, words):
            
            self.rect_predicted[i].set_width(box[2] - box[0])
            self.rect_predicted[i].set_height(box[3] - box[1])
            self.rect_predicted[i].set_xy((box[0], box[1]))
            self.rect_predicted[i].set_label(word)
            new_color = self.start_color + (self.end_color - self.start_color) * float(i + 1) / (len(boxes))
            self.rect_predicted[i].set_edgecolor(new_color)
        for i in range(len(boxes), self.max_timesteps):
            self.rect_predicted[i].set_width(0)
            self.rect_predicted[i].set_height(0)
            self.rect_predicted[i].set_xy((0, 0))
            self.rect_predicted[i].set_label('')
            
        self.ax.legend(loc='best',prop={'size':22})
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
    args['caffemodel'] = 'output/faster_rcnn_end2end/vg_1.0_train/faster_rcnn_cap_two_stage_context8_finetune3_iter_200000.caffemodel'
    
    args['feature_prototxt'] = 'models/faster_rcnn_cap/vgg_region_global_feature_given_box.prototxt'
    args['embed_prototxt'] = 'models/faster_rcnn_cap/test_word_embedding_512.prototxt'
    args['recurrent_prototxt'] = 'models/faster_rcnn_cap/test_cap_pred_context8.prototxt'
    feature_net = caffe.Net(args['feature_prototxt'], args['caffemodel'], caffe.TEST)
    embed_net = caffe.Net(args['embed_prototxt'], args['caffemodel'], caffe.TEST)
    recurrent_net = caffe.Net(args['recurrent_prototxt'], args['caffemodel'], caffe.TEST)
    max_timesteps = 15
    cfg.TEST.HAS_RPN=False
    cfg.MAX_SIZE = 720

    a = Interface(im_path, im_list, vocab_path, feature_net, embed_net, recurrent_net, max_timesteps)
    plt.show()

if __name__ =='__main__':
    main()