from scipy.signal import resample
from golem import DataSet

class Resample:
  def __init__(self, factor=None, axis=0):
    assert axis > 0, 'Cannot resample over instances'
    self.factor = factor
    self.axis = axis

  def train(self, d): pass

  def test(self, d):
    assert self.axis < d.nd_xs.ndim
    new_len = self.factor * d.nd_xs.shape[self.axis]
    xs = resample(d.nd_xs, new_len, axis=self.axis)
    feat_shape = xs.shape[1:]
    return DataSet(xs=xs.reshape(d.ninstances, -1), feat_shape=feat_shape, 
      default=d)
