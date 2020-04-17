import numpy as np
from collections import deque
import tensorflow as tf

class Histogram():

    def __init__(self, size, dimensions, tbPrefix):
        self.size = size
        self.dimensions = dimensions
        self.buffers = [deque(maxlen = size) for i in range(dimensions)]
        self.curr = 0
        self.prefix = tbPrefix
        self.phs = []
        self.summaries = []
        
        for i in range(dimensions):
            phCurr = tf.placeholder(dtype=tf.float32, shape=[None], name="{}_ph{}".format(self.prefix,i))
            self.phs.append(phCurr)
            self.summaries.append(tf.summary.histogram("{}_histogram/{}".format(self.prefix,i), phCurr))
    
    def addValue(self, value):
        assert(len(value.shape) == 2 and value.shape[1]==self.dimensions)
        for i in range(self.dimensions):
            self.buffers[i].append(value[0,i])
        if self.curr < self.size:
            self.curr += 1
        
    def getHistogram(self, bins=10, normalized=True):
        hists = []
        bin_edgs = []
        weights=[1/self.curr for i in range(self.curr)] if normalized else None  

        for i in range(self.dimensions):
            hist, bin_edges = np.histogram(self.buffers[i], bins=bins, weights=weights)
            hists.append(hist)
            bin_edgs.append(bin_edges)

        return hists, bin_edgs
    
    def getValues(self):
        return [list(self.buffers[i]) for i in range(self.dimensions)]
        
        

