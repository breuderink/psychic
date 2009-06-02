import numpy as np
from scipy import signal
from golem import DataSet

class Filter:
  def __init__(self, b, a):
    self.b, self.a = b, a

  def train(self, d):
    pass

  def test(self, d):
    assert len(d.feat_shape) == 1 # signal.lfiler can crash on 2D data
    fxs = signal.lfilter(self.b, self.a, d.xs, axis=0)
    return DataSet(xs=fxs.reshape(d.xs.shape), default=d)

class FBFilter:
  def __init__(self, b, a, axis=0):
    self.b, self.a = b, a

  def train(self, d):
    pass

  def test(self, d):
    assert len(d.feat_shape) == 1 # signal.lfiler can crash on 2D data
    b, a = self.b, self.a
    fs = signal.lfilter(b, a, d.xs, axis=0) # forward
    fs = np.flipud(signal.lfilter(b, a, np.flipud(fs), axis=0)) # backward
    return DataSet(xs=fs.reshape(d.xs.shape), default=d)
