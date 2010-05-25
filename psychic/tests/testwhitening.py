import unittest, logging
import os.path
import numpy as np
import pylab
from ..nodes import Whitening
from golem import DataSet, data

class TestWhitening(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    da = data.gaussian_dataset([100, 100])
    self.d = DataSet(np.hstack([da.xs, np.random.rand(da.ninstances, 6)]), 
      da.ys, da.ids, feat_shape=(2, 4), 
      feat_nd_lab=[['t0', 't1'], ['a', 'b', 'c', 'd']])
    self.n = Whitening()

  def test_nocov(self):
    '''Test that the covariance is the identity matrix'''
    d, n = self.d, self.n
    n.train(d)
    d2 = n.apply(d)

    cov = np.cov(np.concatenate(d2.nd_xs, axis=0), rowvar=False)
    self.assertEqual(cov.shape, (4, 4))
    np.testing.assert_almost_equal(cov, np.eye(4))
    self.assertEqual(d2.ninstances, d.ninstances)
    self.assertEqual(d2.nfeatures, d.nfeatures)
    self.assertEqual(d2.feat_nd_lab, [['t0', 't1'], 
      ['WC%d' % i for i in range(4)]])

  def test_rank(self):
    '''Test that Whitning works with lower rank data'''
    n = self.n
    d = self.d
    d = DataSet(xs=np.tile(d.xs[:, 0].reshape(-1, 1), (1, 8)), default=d)
    n.train(d)
    d2 = n.apply(d)
    self.assertEqual(d2.ninstances, d.ninstances)
    self.assertEqual(d2.feat_shape, (2, 1))
    self.assertEqual(d2.feat_nd_lab, [['t0', 't1'], ['WC0']])
