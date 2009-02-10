import unittest
import numpy as np
from psychic.bcihelpers import *

class TestSlidingWindow(unittest.TestCase):
  def test_functionality(self):
    signal = np.arange(10)
    windows = sliding_window(signal, 5, 2)
    self.assertEqual(windows.shape, (3, 5))
    np.testing.assert_equal(windows[:, 0], [0, 2, 4])
    np.testing.assert_equal(windows[0, :], range(5))

class TestSpectrogram(unittest.TestCase):
  def test_not_implemented(self):
    self.beta = np.sin(np.linspace(0, 30 * 2 * np.pi, 512))
    #@@FIXMEself.assert_(False)

class TestSlice(unittest.TestCase):
  def setUp(self):
    self.frames = np.arange(40).reshape(-1, 2)

  def test_slice(self):
    slices = slice(self.frames, [2, 16], pre_frames=2, post_frames=4)
    self.assertEqual(slices.shape, (2, 6, 2))
    np.testing.assert_equal(slices[0, :, :], np.arange(0, 12).reshape(-1, 2))
    np.testing.assert_equal(slices[1, :, :], np.arange(28, 40).reshape(-1, 2))

  def test_outside(self):
    frames = np.arange(40).reshape(-1, 2)
    self.assertRaises(Exception, slice, 
      self.frames, [1, 19], pre_frames=2, post_frames=4)
