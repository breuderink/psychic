import unittest
import numpy as np
from psychic.preprocessing import *

class TestEventDetection(unittest.TestCase):
  def setUp(self):
    pass

  def do_test(self, status, events, indices):
    e, i = status_to_events(status)
    np.testing.assert_equal(e, events)
    np.testing.assert_equal(i, indices)

  def test_single_event(self):
    self.do_test([0, 0, 0, 1, 1, 0, 0, 0], [1], [3])

  def test_functionality(self):
    self.do_test([2, 0, 1, 1, 3, 0, 4], [2, 1, 3, 4], [0, 2, 4, 6])

  def test_degenerate(self):
    self.do_test([0], [], [])

class TestCAR(unittest.TestCase):
  def test_CAR(self):
    d = np.random.rand(20, 3)
    d2 = car(d)
    np.testing.assert_almost_equal(d2.mean(axis=1), np.zeros(d.shape[0]))

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
    #@@FIXME
    self.assert_(False)

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
