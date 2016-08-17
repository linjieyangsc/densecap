import _init_paths
import caffe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from fast_rcnn.test_cap_manual_proposal import region_captioning, sentence
from fast_rcnn.config import cfg
import cv2

class Interface(object):
    def __init__(self, image_path, image_list, vocabulary_path, models):
        self.ax = plt.axes([0,0,1,1])
        self.colors = ['g','r','c','m','y','k']
        # init rectangles to be drawn
        self.rect = Rectangle((0,0), 0, 0, fill=False, edgecolor='b',linewidth=2)
        self.rect_predicted = []
        self.model_n = len(models)
        self.models = models
        for i in xrange(self.model_n):
            self.rect_predicted.append(Rectangle((0,0), 0, 0, fill=False, edgecolor=self.colors[i],linewidth=2))
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
            for i in range(self.model_n):
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
        strategy = {'algorithm': 'greedy'}
        for i, model in zip(range(self.model_n), self.models):
            boxes, captions, _ = region_captioning(model['feature_net'], model['embed_net'], model['recurrent_net'], self.im, box, strategy)
            #add new box
            box_pred = boxes[0,:]
            caption = captions[0]
            caption_str = sentence(self.vocab, caption)
            self.rect_predicted[i].set_width(box_pred[2] - box_pred[0])
            self.rect_predicted[i].set_height(box_pred[3] - box_pred[1])
            self.rect_predicted[i].set_xy((box_pred[0], box_pred[1]))
            self.rect_predicted[i].set_label(caption_str)
     
            
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
    models=[]
    model_names = ('two_stage4_512_finetune','two_stage_context8_finetune3')
    recurrent_proto_names = ('test_cap_pred4_512','test_cap_pred_context8')
    feature_proto_names = ('vgg_region_feature_given_box','vgg_region_global_feature_given_box')
    for model_name, feature_proto_name, recurrent_proto_name in zip(model_names, feature_proto_names, recurrent_proto_names):
        model = {}
        caffemodel = 'output/faster_rcnn_end2end/vg_train/faster_rcnn_cap_%s_iter_200000.caffemodel' \
            % model_name
        feature_prototxt = 'models/faster_rcnn_cap/%s.prototxt' % feature_proto_name
        embed_prototxt = 'models/faster_rcnn_cap/test_word_embedding_512.prototxt'
        recurrent_prototxt = 'models/faster_rcnn_cap/%s.prototxt' % recurrent_proto_name

        
        model['feature_net'] = caffe.Net(feature_prototxt, caffemodel, caffe.TEST)
        model['embed_net'] = caffe.Net(embed_prototxt, caffemodel, caffe.TEST)
        model['recurrent_net'] = caffe.Net(recurrent_prototxt, caffemodel, caffe.TEST)
        models.append(model)
    #global config parameters
    cfg.TEST.HAS_RPN=False
    cfg.MAX_SIZE = 720

    a = Interface(im_path, im_list, vocab_path, models)
    plt.show()

if __name__ =='__main__':
    main()