import logging
import numpy as np
from golem import DataSet

class Whitening:
  def __init__(self):
    pass

  def train(self, d):
    assert len(d.feat_shape) == 2 # only work with 2D features, time*channels
    
    # Store mean
    xs = np.concatenate(d.nd_xs, axis=0)
    self.mean = np.mean(xs, axis=0)

    # Calc whitening matrix P
    cov = np.cov(xs, rowvar=False) 
    U, s, V = np.linalg.svd(cov)
    P = np.dot(U, np.linalg.pinv(np.diag(s)) ** (.5))
    rank = np.sum(s > 1e-8)
    self.P = P[:, :rank]

  def test(self, d):
    xs = np.concatenate(d.nd_xs, axis=0) - self.mean
    xs = np.dot(xs, self.P).reshape(d.ninstances, -1)
    feat_shape = (d.feat_shape[0], self.P.shape[1])
    if d.feat_nd_lab:
      feat_nd_lab = [d.feat_nd_lab[0], 
        ['WC%d' % ci for ci in range(feat_shape[1])]]
    else:
      feat_nd_lab = None
    return DataSet(xs, feat_shape=feat_shape, feat_nd_lab=feat_nd_lab, 
      default=d)
