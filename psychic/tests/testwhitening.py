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
    self.d = DataSet(np.hstack([da.xs, np.random.random(da.xs.shape)]), 
      da.ys, da.ids, feat_shape=(1, 4), 
      feat_nd_lab=[['f0'], ['a', 'b', 'c', 'd']])
    self.n = Whitening()

  def test_nocov(self):
    '''Test that the covariance is the identity matrix'''
    d, n = self.d, self.n
    n.train(d)
    d2 = n.test(d)

    cov = np.cov(d2.xs, rowvar=False)
    self.assertEqual(cov.shape, (4, 4))
    np.testing.assert_almost_equal(cov, np.eye(4))
