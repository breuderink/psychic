import unittest
import numpy as np
from golem import DataSet
from ..nodes import RegFilter

import pylab

class TestRegFilter(unittest.TestCase):
  NINSTANCES = 400

  def setUp(self):
    np.random.seed(2)
    self.eog = np.random.randn(TestRegFilter.NINSTANCES, 4)
    self.eeg = np.random.randn(TestRegFilter.NINSTANCES, 8)

    self.eeg -= np.mean(self.eeg, axis=0)
    self.eog -= np.mean(self.eog, axis=0)

    self.mix_m = np.random.randn(4, 8)
    self.signal = self.eeg + 10 * np.dot(self.eog, self.mix_m)

  def test_removal(self):
    d = DataSet(xs=np.hstack([self.signal, self.eog]), 
      ys=np.zeros((TestRegFilter.NINSTANCES, 1)))
    n = RegFilter(np.arange(12) >= self.eeg.shape[1])
    n.train(d)
    d2 = n.test(d)

    self.assertEqual(d2.nfeatures, 8)
    self.assertEqual(d2.ninstances, d.ninstances)

    # before
    cross_cov = np.cov(np.hstack([d.xs, self.eog]), rowvar=False)
    self.assert_((cross_cov[:-4, -4:] != 0).all())

    # after
    cross_cov = np.cov(np.hstack([d2.xs, self.eog]), rowvar=False)
    np.testing.assert_almost_equal(cross_cov[:-4, -4:], 0)
    np.testing.assert_almost_equal(d2.xs, self.eeg, 0)

  def test_feat_lab(self):
    d = DataSet(xs=np.hstack([self.signal, self.eog]), 
      ys=np.zeros((TestRegFilter.NINSTANCES, 1)),
      feat_lab=['f%d' % fi for fi in range(12)])
    mask = np.arange(12) % 2
    n = RegFilter(mask)
    n.train(d)
    d2 = n.test(d)
    self.assertEqual(d2.feat_lab, 
      ['f%d' % fi for fi in range(12) if not fi % 2])

    self.assertEqual(d2.nfeatures, 6)
    self.assertEqual(d2.ninstances, d.ninstances)
