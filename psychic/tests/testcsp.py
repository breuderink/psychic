import unittest, logging
import os.path
import numpy as np
import pylab
from ..nodes import CSP
from golem import DataSet, data, plots

class TestCSP(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    da = data.gaussian_dataset([100, 100])
    self.d = DataSet(np.hstack([da.xs, np.random.random(da.xs.shape)]), 
      da.ys, da.ids, feat_shape=(1, 4))
    self.n = CSP(m=2)

  def test_class_diag_descending(self):
    '''Test for diagonal, descending class cov'''
    d, n = self.d, self.n
    n.train(d)
    d2 = n.test(d.get_class(0))

    self.assertEqual(d2.nfeatures, 2)
    self.assertEqual(d2.nclasses, 2)

    cov = np.cov(d2.xs, rowvar=False)
    self.assertAlmostEqual(np.trace(cov), np.sum(cov))
    np.testing.assert_equal(np.diag(cov), np.sort(np.diag(cov))[::-1])

  def test_nocov(self):
    '''Test that the CSP-transformed features are uncorrelated'''
    d, n = self.d, self.n
    n.train(d)
    d2 = n.test(d)

    cov = np.cov(d2.xs, rowvar=False)
    np.testing.assert_almost_equal(cov, np.eye(2))

  def test_m(self):
    '''Test that CSP selects the right number of components'''
    logging.getLogger('golem.CSP').setLevel(logging.ERROR)

    d = DataSet(xs=np.hstack([self.d.xs] * 2), feat_shape=(1, 8), 
      default=self.d)

    for m in [2, 4, 6]:
      n = CSP(m=m)
      n.train(d)
      d2 = n.test(d)

      if m <= 4:
        self.assertEqual(d2.nfeatures, m)
      else:
        self.assertNotEqual(d2.nfeatures, m)

    logging.getLogger('golem.CSP').setLevel(logging.WARNING)
  
  def test_plot(self):
    '''Plot CSP for visual inspection'''
    d, n = self.d, self.n
    n.train(d)
    d2 = n.test(d)
    plots.scatter_plot(d2)
    pylab.savefig('csp.eps')
    pylab.close()
  
  def test_2d(self):
    '''Test CSP on 2D-shaped features'''
    d = DataSet(self.d.xs.reshape(-1, 8), self.d.ys[::2], self.d.ids[::2], 
      feat_shape=(2, 4))
    n = CSP(m=2)
    n.train(d)
    d = n.test(d)
    self.assertEqual(d.nfeatures, 4)
    self.assertEqual(d.nclasses, 2)

    cov = np.cov(np.concatenate(d.nd_xs, axis=0), rowvar=False)
    np.testing.assert_almost_equal(cov, np.eye(cov.shape[0]))
