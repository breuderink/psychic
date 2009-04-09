import unittest, os
import numpy as np
from ..nodes import BDFFile

import pylab
class TestBDF(unittest.TestCase):
  def test_load(self):
    b = BDFFile(os.path.join('data', 'sine-256Hz.bdf'))
    d = b.train()

    # test labels
    targets = ['A%d' % (i + 1) for i in range(16)]
    self.assertEqual(d.feat_lab, targets)
    self.assertEqual(d.cl_lab, ['status'])

    # test ids ~ time
    self.assertAlmostEqual(d.ids[256 + 1], 1, 2)

    # test dims
    self.assertEqual(d.nfeatures, 16)
    self.assertEqual(d.ninstances, 60 * 256)
