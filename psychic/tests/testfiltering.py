import unittest, os
import numpy as np
import pylab
from scipy import signal
from golem import DataSet
from ..plots import plot_filt_char, plot_timeseries
from ..filtering import *

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


class TestResample(unittest.TestCase):
  def setUp(self):
    xs = np.arange(1000).reshape(-1, 2)
    ys = np.zeros((500, 1))
    ys[::4] = 2
    self.d = DataSet(xs=xs, ys=ys)

  def test_resample(self):
    d = self.d
    d2 = resample_rec(d, .5)
    self.assertEqual(d2.ninstances, d.ninstances / 2)
    self.assertEqual(d2.nfeatures, d.nfeatures)
    self.assertEqual(d2.feat_lab, d.feat_lab)
    self.assertEqual(d2.cl_lab, d.cl_lab)
    self.assertEqual(d2.feat_shape, d.feat_shape)
    np.testing.assert_equal(d2.ys[::2], 2)
    self.assertEqual(np.mean(np.diff(d2.ids.flatten())), 2)

    # Testing the decimation itself is more difficult due to boundary
    # artifacts, and is the responsibility of Scipy.
    # We do a rough test that it should be similar to naive resampling:
    self.assert_(np.std((d2.xs - d.xs[::2])[100:-100,:]) < 0.6)

  def test_overlapping_markers(self):
    d = self.d
    # test overlapping markers
    self.assertRaises(AssertionError, decimate_rec, d, 5)


class TestDecimate(unittest.TestCase):
  def setUp(self):
    xs = np.arange(100).reshape(-1, 2)
    ys = np.zeros((50, 1))
    ys[::4] = 2
    self.d = DataSet(xs=xs, ys=ys)

  def test_aa(self):
    # Create signal with a LF and a HF part. HF should cause aliasing
    xs = np.zeros(128)
    xs[[-2, -3]] = 4 # HF
    xs[8] = 1 # LF
    xs = np.fft.irfft(xs).reshape(-1, 1) + 1

    ys = np.zeros(xs.shape)
    ys[::4] = 2
    d = DataSet(xs=xs, ys=ys)

    d2 = decimate_rec(d, 2)
    self.assertEqual(d2.ninstances, d.ninstances / 2)
    self.assertEqual(d2.nfeatures, d.nfeatures)
    self.assertEqual(d2.feat_lab, d.feat_lab)
    self.assertEqual(d2.cl_lab, d.cl_lab)
    self.assertEqual(d2.feat_shape, d.feat_shape)
    np.testing.assert_equal(d2.ys[::2], 2)
    self.assertEqual(np.mean(np.diff(d2.ids.flatten())), 2)

    self.assertEqual(np.argsort(np.abs(np.fft.rfft(d.xs[::2,0])))[-2],
      2, 'Without the AA-filter the f=1./2 has most power')
    self.assertEqual(np.argsort(np.abs(np.fft.rfft(d2.xs[:,0])))[-2],
      8, 'With the AA-filter, f=1./8 has most power.')

  def test_decimate(self):
    d = self.d
    d2 = decimate_rec(d, 2)
    self.assertEqual(d2.ninstances, d.ninstances / 2)
    self.assertEqual(d2.nfeatures, d.nfeatures)
    self.assertEqual(d2.feat_lab, d.feat_lab)
    self.assertEqual(d2.cl_lab, d.cl_lab)
    self.assertEqual(d2.feat_shape, d.feat_shape)
    np.testing.assert_equal(d2.ys[::2], np.ones((13, 1)) * 2)
    self.assertEqual(np.mean(np.diff(d2.ids.flatten())), 2)

  def test_overlapping_markers(self):
    d = self.d
    # test overlapping markers
    self.assertRaises(AssertionError, decimate_rec, d, 4)

