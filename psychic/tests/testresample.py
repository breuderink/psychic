import unittest
import numpy as np
from golem import DataSet
from ..nodes import Resample

class TestResample(unittest.TestCase):
  def setUp(self):
    xs = (np.array([0, 1, 0, 1, 0, 1, 0, 1]).reshape(-1, 1) + np.arange(5)
      ).reshape(2, -1)
    ys = np.zeros((2, 1))
    self.d = DataSet(xs=xs, ys=ys, feat_shape=(4, 5))

  def test_lowpass(self):
    d = self.d
    n = Resample(axis=1, factor=.5)
    d2 = n.test(d)

    self.assertEqual(d2.feat_shape, (2, 5))
    self.assertEqual(d2.ninstances, 2)
    
    # resampling in axis=1 should result in a constant signal
    np.testing.assert_equal(
      np.concatenate(d2.nd_xs, axis=0).var(axis=0), np.zeros(5))

  def test_axes(self):
    d = self.d
    n = Resample(axis=2, factor=3/5.)
    d2 = n.test(d)

    self.assertEqual(d2.feat_shape, (4, 3))
    self.assertEqual(d2.ninstances, 2)
    
    # resampling introduces some artifacts, see:
    # pylab.plot(resample(np.linspace(0, 1, 1000), 40))
    regr = np.array(
      [[[ 1., 1.30801829, 3.69198171],
      [ 2., 2.30801829, 4.69198171],
      [ 1., 1.30801829, 3.69198171],
      [ 2., 2.30801829, 4.69198171]],

      [[ 1., 1.30801829, 3.69198171],
      [ 2., 2.30801829, 4.69198171],
      [ 1., 1.30801829, 3.69198171],
      [ 2., 2.30801829, 4.69198171]]])

    np.testing.assert_almost_equal(d2.nd_xs, regr)

  def test_wrong_axis(self):
    self.assertRaises(AssertionError, Resample, axis=0, factor=1)
