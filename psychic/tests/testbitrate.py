import unittest
from ..bitrate import *

class TestBitrate(unittest.TestCase):
  def test_wolpaw(self):
    self.assertAlmostEqual(wolpaw_bitr(2, 1/2.), 0)
    self.assertAlmostEqual(wolpaw_bitr(2, 1), 1)
    self.assertAlmostEqual(wolpaw_bitr(2, 0), 1)
    self.assertAlmostEqual(wolpaw_bitr(3, 1/3.), 0)
    self.assertAlmostEqual(wolpaw_bitr(4, 1/4.), 0)
    self.assertAlmostEqual(wolpaw_bitr(4, 1), 2)
