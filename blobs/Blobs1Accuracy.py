import caffe
import numpy as np
import sys

class Blobs1Accuracy(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to check equality.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
		self.diff = np.transpose(bottom[0].data)- bottom[1].data
		self.diff = np.where(self.diff<0.0,  -self.diff, self.diff)
		self.diff = self.diff-0.5
		
		#print >> sys.stderr, np.transpose(bottom[0].data) , bottom[1].data, self.diff
		
		top[0].data[0] = np.average( self.diff <= 0.0 )
		
    def backward(self, top, propagate_down, bottom):
    	# forward only layer
        return
