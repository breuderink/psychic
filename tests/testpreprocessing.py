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
