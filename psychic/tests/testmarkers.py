import unittest
import numpy as np
from .. import markers

class TestMarkerDetection(unittest.TestCase):
  def setUp(self):
    pass

  def check(self, status, events, indices):
    e, ei = markers.markers_to_events(status)
    np.testing.assert_equal(events, e)
    np.testing.assert_equal(indices, ei)

  def test_single_event(self):
    self.check([0, 0, 0, 1, 1, 0, 0, 0], [1], [3])

  def test_functionality(self):
    self.check([2, 0, 1, 1, 3, 0, 4], [2, 1, 3, 4], [0, 2, 4, 6])

  def test_degenerate(self):
    self.check([0], [], [])

class TestResampleStatus(unittest.TestCase):
  def test_normal(self):
    np.testing.assert_equal(
      markers.resample_markers([1, 0, 0, 2, 0, 0], 2), [1, 2])
    np.testing.assert_equal(
      markers.resample_markers([0, 0, 0, 0, 0, 0], 2), [0, 0])
    np.testing.assert_equal(
      markers.resample_markers([1, 0, 0, 0, 0, 2], 2), [1, 2])
    np.testing.assert_equal(
      markers.resample_markers([0, 0, 1, 0, 0, 2], 2), [1, 2])
    np.testing.assert_equal(
      markers.resample_markers([0, 0, 1, 0, 0, 2], 3), [0, 1, 2])

  def test_overlap(self):
    self.assertRaises(AssertionError, markers.resample_markers,
      [1, 2, 3, 4], 2)
    self.assertRaises(AssertionError, markers.resample_markers,
      [0, 0, 3, 4], 2)
    self.assertRaises(AssertionError, markers.resample_markers,
      [1, 2, 0, 0], 2)

  def test_overlap_delay(self):
    np.testing.assert_equal(
      markers.resample_markers([1, 2, 0, 0], 2, max_delay=1), [1, 2])
    np.testing.assert_equal(
      markers.resample_markers([1, 2, 3, 0, 0, 0], 3, max_delay=1), [1, 2, 3])
    self.assertRaises(AssertionError, markers.resample_markers,
      [1, 2, 3, 4, 0, 0, 0, 0], 4, max_delay=1)
    self.assertRaises(AssertionError, markers.resample_markers,
      [1, 2, 3, 0], 2, max_delay=2)

  def test_spacing(self):
    np.testing.assert_equal(
      markers.resample_markers([1, 0, 1, 0, 0, 0], 3, max_delay=1), [1, 0, 1])

class TestGhostMarkers(unittest.TestCase):
  def test_detection(self):
    gm = markers.biosemi_find_ghost_markers
    np.testing.assert_equal(
      gm([1, 1, 0, 2, 2, 2|4, 4, 0, 0, 2, 2|5, 5, 0]), [5, 10])
    np.testing.assert_equal(gm([1, 1|4, 4]), [1])
    np.testing.assert_equal(gm([1, 0, 1|4, 0, 4]), [])
    np.testing.assert_equal(gm([1, 0, 1|4, 4]), [])
    np.testing.assert_equal(gm([1, 1|4, 0, 4]), [])
    np.testing.assert_equal(gm([1, 3, 4]), [])

    # cannot be detected, but that does not matter:
    np.testing.assert_equal(gm([1, 1|3, 3]), [])

  def test_ghost_too_long(self):
    gm = markers.biosemi_find_ghost_markers
    np.testing.assert_equal(gm([1, 1|4, 1|4, 4]), [])

  def test_empty(self):
    gm = markers.biosemi_find_ghost_markers
    np.testing.assert_equal(gm(np.zeros(10)), [])


