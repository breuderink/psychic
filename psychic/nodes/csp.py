import logging
import numpy as np
from golem import DataSet

class CSP:
  def __init__(self, m=2):
    assert(m % 2 == 0)
    self.W = None
    self.m = m

  def train(self, d):
    assert(d.nclasses == 2)
    assert(len(d.feat_shape) == 2) # only work with 2D features, time*channels
    
    # Store mean
    xs = np.concatenate(d.nd_xs, axis=0)
    self.mean = np.mean(xs, axis=0)

    # Calc whitening matrix P
    cov = np.cov(xs, rowvar=False) 
    U, s, V = np.linalg.svd(cov)
    P = np.dot(U, np.linalg.pinv(np.diag(s)) ** (.5))
    rank = np.sum(s > 1e-8)
    self.P = P[:, :rank]
    
    # Calc class-diagonalization matrix B
    d0 = d.get_class(0)
    xs0 = np.dot(np.concatenate(d0.nd_xs, axis=0) - self.mean, self.P)
    self.B, s0, V0 = np.linalg.svd(np.cov(xs0, rowvar=False))

    # Construct final transformation matrix W
    self.W = np.dot(self.P, self.B)
    if self.W.shape[1] >= self.m:
      comps = range(self.m / 2) + range(-self.m / 2, 0)
      logging.getLogger('golem.CSP').debug('Selecting components %s' % comps)
      self.W = self.W[:, comps]
    else:
      logging.getLogger('golem.CSP').warning(
        'Rank to low to select %d components. W.shape = %s' %
        (self.m, self.W.shape))
      self.m = self.W.shape[1]

    self.cl_lab = d.cl_lab

  def test(self, d):
    assert(len(d.feat_shape) == 2)
    xs = np.concatenate(d.nd_xs, axis=0) - self.mean
    xs = np.dot(xs, self.W).reshape(d.ninstances, -1)
    feat_shape = (d.feat_shape[0], self.m)
    if d.feat_nd_lab:
      feat_nd_lab = [d.feat_nd_lab[0], ['CSP%d' % ci for ci in range(self.m)]]
    else:
      feat_nd_lab = None
    return DataSet(xs, feat_nd_lab=feat_nd_lab, feat_shape=feat_shape, 
      cl_lab=self.cl_lab, default=d)

  def __str__(self):
    W = self.eigen_cols
    if W == None:
      return 'CSP (untrained)'
    return 'CSP (%dD -> %dD)' % (W.shape[0], m)
