import numpy as np
from scipy import signal
from golem import DataSet

class FilterNode:
  def __init__(self, b, a, axis=0):
    self.b, self.a = b = b, a
    self.axis = axis

  def train(self, d):
    return self.test(d)

  def test(self, d):
    fxs = signal.lfilter(self.b, self.a, d.nd_xs, axis=self.axis)
    return DataSet(xs=fxs.reshape(d.xs.shape), default=d)
