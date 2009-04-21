import unittest
import numpy as np
from golem import DataSet
from .. import nodes

class TestChannVar(unittest.TestCase):
  def setUp(self):
    xs = (np.arange(10).reshape(-1, 1) * [-1, 1, -2, 2]).reshape(2, 5, 4)
    ys = np.zeros((2, 1))
    self.d = DataSet(xs.reshape(2, -1), ys, feat_shape=(5, 4))

  def test(self):
    d = self.d
    d2 = nodes.ChannVar().test(d)
    np.testing.assert_equal(d2.nd_xs[:, :2], 2 * np.ones((2, 2)))
    np.testing.assert_equal(d2.nd_xs[:, 2:], 8 * np.ones((2, 2)))
