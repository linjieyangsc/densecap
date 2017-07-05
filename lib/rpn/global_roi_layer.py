##################################################
# This layer only applies to single object class
# Especially for faster-rcnn-lstm since labels no longer denote the object classes 
# Linjie Yang, Chinese University of Hong Kong
# 04/21/2016
###################################################
import caffe
import yaml
import numpy as np
import numpy.random as npr

class GlobalRoILayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        #only one roi (global roi)
        top[0].reshape(1, 5)
        
    def forward(self, bottom, top):
        
        im_info = bottom[0].data[0,:]
        
        
        top[0].reshape(1, 5)
        top[0].data[...] = np.array([0, 0, 0, im_info[1]-1, im_info[0]-1])

        
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

