# This python layer aims to check the consistency of bbox regression targets
# Linjie Yang, Chinese University of Hong Kong
import caffe
import numpy as np
import yaml
import cPickle
class DebugBBoxRegLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""



    def setup(self, bottom, top):
        """Setup the DebugBBoxRegLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._time_steps = layer_params['time_steps']
        assert(bottom[0].data.shape[0] == self._time_steps)
        assert(bottom[1].data.shape[0] == self._time_steps)
        assert(bottom[1].data.shape == bottom[0].data.shape)
        


    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        
        #bottom[0].data==0 --> bottom[1].data==0
        
        #check repeativity of bottom[0], bottom[1]
        s2 = np.sum(bottom[0].data[0,:,:] != bottom[0].data[1,:,:])
        assert(s2 == 0)
        s3 = np.sum(bottom[0].data[0,:,:] != bottom[0].data[10,:,:])
        assert(s3 == 0)
        s4 = np.sum(bottom[1].data[:,:,0] != bottom[1].data[:,:,1])
        assert(s4 == 0)
        s0 = np.sum(bottom[1].data[:,:,0], axis = 0) > 0 # cont_sentence
        s1 = np.sum(np.sum(bottom[0].data[:,:,:], axis = 0), axis = 1) != 0 # bbox_target

        inconsist_n = np.sum(s0[s1]==0)
        assert(inconsist_n == 0)
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    

