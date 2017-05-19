import caffe
import numpy as np
import sys

class Blobs1Loss(caffe.Layer):
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
		
		c = 0.333
		
		# absolute
		self.diff = np.where((self.diff<0.0) & (self.diff>-c) == True, 0., self.diff)
		self.diff = np.where((self.diff>0.0) & (self.diff< c) == True, 0., self.diff)
		self.diff = np.where(self.diff>0.0, self.diff-c, self.diff)
		self.diff = np.where(self.diff<0.0, self.diff+c, self.diff)
		#self.diff = np.where(self.diff<0.0, self.diff, self.diff)
		
		# subctracting non relevant error
		#self.diff = self.diff-0.5
		
		top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.
		
		#print >> sys.stderr, np.transpose(bottom[0].data) , bottom[1].data, self.diff, top[0].data
		
    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * np.transpose(self.diff) / bottom[i].num
    