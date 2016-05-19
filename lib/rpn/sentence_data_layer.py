##################################
# This python layer accepts region ids as input and retrieves region sentences for them.
# Linjie Yang, Chinese University of Hong Kong
# 04/21/2016
###############################
import caffe
import numpy as np
import yaml
import pprint
import cPickle
from collections import Counter
DEBUG=False
class SentenceDataLayer(caffe.Layer):
    """This python layer accepts region ids as input and retrieves region sentences for them."""



    def setup(self, bottom, top):
        """Setup the SentenceDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        all_modes = ('repeat','concat')
        self._time_steps = layer_params['time_steps']
        phrase_path = layer_params['phrase_path']
        self._mode = layer_params['mode'] if 'mode' in layer_params else 'repeat'
        assert (self._mode in all_modes)
        self._all_phrases = cPickle.load(open(phrase_path,'rb'))
        if DEBUG:
            all_len = [len(stream) for k,stream in self._all_phrases.iteritems()]
            count_len = Counter(all_len)
            print count_len
        #all_regions is a dict from region id to caption stream
        assert(len(bottom) == 1) #only one bottom: labels (region ids)
        num_regions = bottom[0].data.shape[0]
        
        for i in xrange(len(top)):
         
            top[i].reshape(self._time_steps, num_regions)
            if self._mode == 'concat':
                top[0].reshape(self._time_steps-1, num_regions)
        


    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        assert(len(bottom) == 1) #only one bottom: labels (region ids)
        num_regions = bottom[0].data.shape[0]
        #print num_regions
        if self._mode == 'repeat':
            top[0].reshape(self._time_steps,num_regions)
        else:
            top[0].reshape(self._time_steps-1,num_regions)
        top[1].reshape(self._time_steps,num_regions)
        top[2].reshape(self._time_steps,num_regions)
        if (len(top) > 3):
            top[3].reshape(self._time_steps, num_regions)
        for i in xrange(num_regions):
            stream = self._get_streams(int(bottom[0].data[i]))
            top[0].data[:,i] = stream['input_sentence']
            top[1].data[:,i] = stream['target_sentence']
            top[2].data[:,i] = stream['cont_sentence']
            if (len(top) > 3):
                top[3].data[:,i] = stream['cont_bbox']
        if DEBUG:
            print 'sentence data layer input (first 3)'
            print bottom[0].data[:3]
            print 'sentence data layer output (first 3)'
            print top[0].data[:,:3]
            print top[1].data[:,:3]
            print top[2].data[:,:3]
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    

    def _get_streams(self, region_id):

        if self._mode =='repeat':
            # Image feature repeated at each time step
            if region_id > 0:
                stream = self._all_phrases[region_id]
                pad = self._time_steps - (len(stream) + 1) 
                out = {}
                out['cont_sentence'] = [0] + [1] * len(stream) + [0] * pad
                out['input_sentence'] = [0] + stream + [-1] * pad
                out['target_sentence'] = stream + [0] + [-1] * pad
                # only make prediction at the last time step for bbox
                out['cont_bbox'] = [0] * len(stream) + [1] + [0] * pad
                #print "target coord length %d" % len(out['target_coord'])
                
                for key, val in out.iteritems():
                    if len(val) > self._time_steps:                    
                        out[key] = val[:self._time_steps]
            else:
                # negative sample, no phrase related
                out = {}
                out['cont_sentence'] = [0] * self._time_steps
                out['input_sentence'] = [-1] * self._time_steps
                out['target_sentence'] = [-1] * self._time_steps
                out['cont_bbox'] = [0] * self._time_steps

        else:
            # Image feature concated to the first time step
            if region_id > 0:
                stream = self._all_phrases[region_id]
                pad = self._time_steps - (len(stream) + 2) 
                out = {}
                out['cont_sentence'] = [0] + [1] * (len(stream) + 1) + [0] * pad
                out['input_sentence'] = [0] + stream + [-1] * pad
                out['target_sentence'] = [-1] + stream + [0] + [-1] * pad
                # only make prediction at the last time step for bbox
                out['cont_bbox'] = [0] * (len(stream) + 1) + [1] + [0] * pad
                #print "target coord length %d" % len(out['target_coord'])
                
                for key, val in out.iteritems():
                    if len(val) > self._time_steps:                    
                        out[key] = val[:self._time_steps]
            else:
                # negative sample, no phrase related
                out = {}
                out['cont_sentence'] = [0] * self._time_steps
                out['input_sentence'] = [-1] * (self._time_steps - 1)
                out['target_sentence'] = [-1] * self._time_steps
                out['cont_bbox'] = [0] * self._time_steps
        return out