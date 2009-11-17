import unittest, operator
import golem
import numpy as np
from ..utils import sliding_window
from ..nodes import SlidingWindow, OnlineSlidingWindow
from scipy import signal

class TestWindowNode(unittest.TestCase):
  def setUp(self):
    xs = np.arange(300).reshape(-1, 3)
    ys = np.linspace(0, 3, 100, endpoint=False).astype(int)
    self.d = golem.DataSet(xs=xs, ys=golem.helpers.to_one_of_n(ys))

  def test_sw(self):
    d = self.d
    for wsize in [4, 10]:
      for wstep in [2, 5]:
        for ref_point in [0, .5, 1]:
          sw = SlidingWindow(wsize, wstep, ref_point)
          d2 = sw.test(d)

          # test shapes
          self.assertEqual(d2.feat_shape, (wsize, d.nfeatures))

          # test xs
          offset = wstep * d.nfeatures
          max_offset = (d.ninstances - wsize) * d.nfeatures + 1
          base_xs = np.arange(0, max_offset, offset)
          detail_xs = np.arange(d.nfeatures * wsize).reshape(1, 
            wsize, d.nfeatures)
          target = base_xs.reshape(-1, 1, 1) + detail_xs
          np.testing.assert_equal(target, d2.nd_xs)

          # test ids
          np.testing.assert_equal(np.diff(d2.ids, axis=0), wstep)
          self.assertEqual(d2.ids[0, 0], int((wsize - 1) * ref_point))

          # test ys
          np.testing.assert_equal(d.ys[d2.ids.flatten()], d2.ys)

  def test_osw(self):
    d = self.d
    for wsize in [4, 10]:
      for wstep in [2, 5]:
        sw = SlidingWindow(10, 5)
        osw = OnlineSlidingWindow(10, 5)

        wins = []
        stream = d
        while len(stream) > 0:
          head, stream = stream[:4], stream[4:]
          wins.append(osw.test(head))

        self.assertEqual(sw.test(d), reduce(operator.add, wins))
