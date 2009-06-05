import unittest, os
import numpy as np
import pylab
from scipy import signal
from golem import DataSet
from ..plots import plot_filt_char, plot_timeseries
from ..filtering import fir_bandpass

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
