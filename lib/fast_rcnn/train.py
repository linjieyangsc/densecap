# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
from utils.debug import softmax
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2
DEBUG=False

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net
        # This is a stupid check, disabled temperally
        scale_bbox_params = False #(cfg.TRAIN.BBOX_REG and
                             #cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             #net.params.has_key('bbox_pred'))

        if scale_bbox_params:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        if DEBUG:
            import cPickle

            phrase_path = 'models/dense_cap/h5_data_distill/buffer_100/train_gt_phrases.pkl'
            self._all_phrases = cPickle.load(open(phrase_path,'rb'))
            vocab_path = 'models/dense_cap/h5_data_distill/buffer_100/vocabulary.txt'
            with open(vocab_path,'r') as f:
                self._vocab = [line.strip() for line in f]
            self._vocab.insert(0, '<EOS>')
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if DEBUG:
                # image = self.solver.net.blobs['data'].data[0,:,:,:].transpose(1,2,0).copy()
                # print 'iter %d' % self.solver.iter
                # print 'shape of image'
                # print image.shape
                # im_info = self.solver.net.blobs['im_info'].data
                # print 'im info'
                # print im_info
                # #gt_boxes = self.solver.net.blobs['gt_boxes'].data
                # rois = self.solver.net.blobs['rois'].data.copy()
                # labels = self.solver.net.blobs['labels'].data.copy()
                # #check sentence

                # sentences = self.solver.net.blobs['target_sentence'].data.copy()
                # print 'shape of sentences'
                # print sentences.shape
                # for i in xrange(sentences.shape[1]):
                    
                #     region_id = labels[i]
                #     if region_id > 0:
                #         sentence = sentences[:,i]
                #         sentence = sentence[:np.where(sentence==0)[0][0]]
                #         assert(np.all(self._all_phrases[region_id] == np.array(sentence)))
                #         print 'checked %d' % i
                #     else:
                #         assert(sentences[0,i] == -1)

                #rois_labels = np.hstack((rois[:,1:],labels[:,np.newaxis]))
                #self.vis_regions(image, rois_labels, self.solver.iter)
                if self.solver.iter > 5: 
                    exit()
                bbox_pred =self.solver.net.blobs['bbox_pred'].data
                print 'bbox pred samples'
                print bbox_pred[:,0,:]
                bbox_target =self.solver.net.blobs['bbox_tile_reshape'].data
                print 'bbox target samples'
                print bbox_target[:,0,:]
                cont_tile =self.solver.net.blobs['cont_tile'].data
                print 'cont tile samples'
                print cont_tile[:,0,:]

                # predict_scores = self.solver.net.blobs['predict'].data
                # predict_labels = np.argmax(predict_scores, axis = 2)
                cont_sentence = self.solver.net.blobs['cont_sentence'].data
                print 'cont sentence sample'
                print cont_sentence[:,0]
                # input_sentence = self.solver.net.blobs['input_sentence'].data
                # print 'input labels sample'
                # print input_sentence[:,:2] 
                # print 'predicted labels sample'
                # print predict_labels[:,:2]
                # target_labels = self.solver.net.blobs['target_sentence'].data
                # print 'target labels sample'
                # print target_labels[:,:2] 
                # predict_probs = softmax(predict_scores)
                # predict_logprobs = np.log(predict_probs)
                # target_probs = np.zeros_like(target_labels)
                # loss = 0
                # time_steps = target_labels.shape[0]
                # samples = target_labels.shape[1]
                # for i in xrange(time_steps):
                #     for j in xrange(samples):
                #         if target_labels[i,j] > -1:
                #             loss += predict_logprobs[i,j,target_labels[i,j]]
                #             target_probs[i,j] = predict_probs[i,j,target_labels[i,j]]
                # loss /= np.sum(target_labels > -1)
                # print 'target probs sample'
                # print target_probs[:,:2] 
                # print 'per word loss: %.3f' % loss
                # word_accuracy = np.sum(predict_labels == target_labels)/np.sum(target_labels > -1)
                # print 'word accuracy: %.3f' % word_accuracy
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)
            #if self.solver_param.test_interval>0 and self.solver.iter % self.solver_param.test_interval == 0:

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths
    
    def vis_regions(self, im, regions, iter_n, save_path='debug'):
        """Visual debugging of detections by saving images with detected bboxes."""
        import cv2
        if not os.path.exists(save_path):
                    os.makedirs(save_path)
        mean_values = np.array([[[ 102.9801,  115.9465,  122.7717]]])
        im = im + mean_values #offset to original values


        for i in xrange(len(regions)):
            bbox = regions[i, :4]
            region_id = regions[i,4]
            if region_id == 0:
                continue
            caption = self.sentence(self._all_phrases[region_id])

            im_new = np.copy(im)     
            cv2.rectangle(im_new, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 2)
            cv2.imwrite('%s/%d_%s.jpg' % (save_path, iter_n, caption), im_new)
    def sentence(self, vocab_indices):
        # consider <eos> tag with id 0 in vocabulary
        sentence = ' '.join([self._vocab[i] for i in vocab_indices])
        return sentence

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb

def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    roidb = filter_roidb(roidb)
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths