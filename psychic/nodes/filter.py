import numpy as np
from scipy import signal
from golem import DataSet

class Filter:
  def __init__(self, b, a, axis=0):
    self.b, self.a = b, a
    self.axis = axis

  def train(self, d):
    return self.test(d)

  def test(self, d):
    fxs = signal.lfilter(self.b, self.a, d.nd_xs, axis=self.axis)
    return DataSet(xs=fxs.reshape(d.xs.shape), default=d)

class FBFilter:
  def __init__(self, b, a, axis=0):
    self.b, self.a = b, a
    self.axis = axis

  def train(self, d):
    return self.test(d)

  def test(self, d):
    b, a = self.b, self.a
    s = np.swapaxes(d.nd_xs, 0, self.axis) 
    fs = signal.lfilter(b, a, s, axis=0) # forward
    fs = np.flipud(signal.lfilter(b, a, np.flipud(fs), axis=0)) # backward
    fs = np.swapaxes(fs, 0, self.axis)
    return DataSet(xs=fs.reshape(d.xs.shape), default=d)
