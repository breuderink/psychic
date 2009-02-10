import unittest
import numpy as np
from psychic.bcihelpers import *

class TestSlidingWindow(unittest.TestCase):
  def testFunctionality(self):
    signal = np.arange(10)
    windows = windowize(signal, 5, 2)
    self.assertEqual(windows.shape, (3, 5))
    np.testing.assert_equal(windows[:, 0], [0, 2, 4])
    np.testing.assert_equal(windows[0, :], range(5))
