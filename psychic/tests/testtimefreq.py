import unittest, os.path
from ..utils import slice, spectrogram
from ..nodes import TFC
from golem import DataSet

import pylab
import numpy as np

FS = 256.

class TestTFC(unittest.TestCase):
  def setUp(self):
    xs = np.array([np.sin(i * 4 * 60 * np.linspace(0, np.pi * 2, 60 * FS)) 
      for i in range(16)]).T
    ys = np.zeros((xs.shape[0], 1))
    ys[[1000, 2000, 3000, 4000], :] = 1
    ids = np.arange(xs.shape[0]).reshape(-1, 1) / FS

    self.d = slice(DataSet(xs=xs, ys=ys, ids=ids), {1:'fake'}, [-512, 512])
    
  def test_setup(self):
    d = self.d
    self.assertEqual(d.feat_shape, (1024, 16))
    self.assertEqual(d.nclasses, 1)
    self.assertEqual(d.ninstances, 4)

  def test_tfc(self):
    d = self.d
    w_size, w_step = 64, 32
    tfc = TFC(w_size, w_step)
    tfc.train(d)
    td = tfc.apply(d)

    nwindows = int(np.floor((d.feat_shape[0] - w_size + w_step) / 
      float(w_step)))
    self.assertEqual(td.feat_shape, (nwindows, w_size/2+1, d.feat_shape[1]))
    self.assertEqual(td.nclasses, d.nclasses)
    self.assertEqual(td.ninstances, d.ninstances)

    for ci in range(td.feat_shape[2]):
      a = td.nd_xs[0,:,:,ci]
      b = spectrogram(d.nd_xs[0,:,ci], w_size, w_step)
      np.testing.assert_equal(a, b)
    self.assertEqual(td.feat_dim_lab, ['time', 'frequency', 'channels'])

    time = [float(t) for t in td.feat_nd_lab[0]]
    np.testing.assert_almost_equal(time, 
      np.linspace((-512 + w_step)/FS, (512 - w_size)/FS, len(time)), 1)

    freq = [float(f) for f in td.feat_nd_lab[1]]
    np.testing.assert_almost_equal(freq, np.arange(32 + 1) * 4, 2)

    self.assertEqual(td.feat_nd_lab[2], d.feat_nd_lab[1])
