import unittest, os
import numpy as np
import pylab
from scipy import signal
from golem import DataSet
from ..plots import plot_filt_char, plot_timeseries
from ..filtering import fir_bandpass
from ..nodes import Filter, FBFilter, BDFFile

class TestFilters(unittest.TestCase):
  def setUp(self):
    self.Fs = 256.

  def test_fir_bandpass(self):
    (b, a) = fir_bandpass(8, 30, 4, Fs=self.Fs)
    bands = [(1, 4, False), (8, 28, True), (10, 12, True), (26, 30, True), 
      (34, 50, False)]

    for (f0, f1, pb) in bands:
      chirp = signal.chirp(np.arange(self.Fs * 10) / self.Fs, 
        f0=f0, t1=10, f1=f1)
      bpchirp = signal.lfilter(b, a, chirp)
      gain = bpchirp.var() / chirp.var()
      if pb:
        self.assert_(.9 < gain < 1.)
      else:
        self.assert_(gain < 1e-2)

  def test_plot_filt_char(self):
    (b, a) = fir_bandpass(50, 100, 4, Fs=self.Fs)
    plot_filt_char(b, a, Fs=self.Fs)
    pylab.savefig('fir_bp50_100.eps')
    pylab.close()

class TestFilter(unittest.TestCase):
  def setUp(self):
    self.d = DataSet(xs=np.arange(100).reshape(-1, 10).astype(float), 
      ys=np.ones(10).reshape(-1, 1))

  def test_filter(self):
    d = self.d
    n = Filter([-1], [1]) # changes sign
    d2 = n.test(d)
    np.testing.assert_equal(self.d.nd_xs, -d2.nd_xs)

  def test_filter_axis(self):
    d = self.d
    n = Filter([1, -1], [1], axis=0)
    d2 = n.test(d)
    np.testing.assert_equal(d2.xs[0, :], np.arange(10)) # warming up
    np.testing.assert_equal(d2.xs[1:, :], np.ones((9, 10)) * 10) # constant diff

    n = Filter([1, -1], [1], axis=1)
    d2 = n.test(d)
    np.testing.assert_equal(d2.xs[:, 0], np.arange(10) * 10) # warming up
    np.testing.assert_equal(d2.xs[:, 1:], np.ones((10, 9))) # constant diff

class TestFBFilter(unittest.TestCase):
  def setUp(self):
    self.d = DataSet(xs=np.arange(100).reshape(-1, 10).astype(float), 
      ys=np.ones(10).reshape(-1, 1))

  def test_zero_delay(self):
    d = self.d
    n = FBFilter([0, 0, 1], [1])
    d2 = n.test(d)
    self.assertEqual(d2[:-2], d[:-2])
    np.testing.assert_equal(d2[-2:].xs, np.zeros((2, 10)))

  def test_filter_axis(self):
    d = self.d
    n = FBFilter([1, -1], [1], axis=0)
    d2 = n.test(d)
    np.testing.assert_equal(d2.xs[-1, :], np.ones(10) * 10)
    np.testing.assert_equal(d2.xs[1:-1,:], np.zeros((8, 10)))
    np.testing.assert_equal(d2.xs[0, :], np.arange(10) - 10)
    
    n = FBFilter([1, -1], [1], axis=1)
    d2 = n.test(d)
    np.testing.assert_equal(d2.xs[:, -1], np.ones(10))
    np.testing.assert_equal(d2.xs[:, 1:-1], np.zeros((10, 8)))
    np.testing.assert_equal(d2.xs[:, 0], np.arange(10) * 10 - 1)
