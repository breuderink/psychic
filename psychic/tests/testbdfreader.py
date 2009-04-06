import unittest
import os.path
import numpy as np
from psychic.bdfreader import *

class TestConversion(unittest.TestCase):
  def setUp(self):
    self.know_values = [
      ((0, 0, 0), 0),
      ((1, 0, 0), 1),
      ((255, 255, 127), (1 << 23) - 1),
      ((0, 0, 128), -(1 << 23)),
      ((255, 255, 255), -1)]
    self.ints = [0, 1, (1 << 23) - 1, -(1 << 23), -1]

  def test_list_conversion(self):
    bytes = list(reduce(lambda x, y: x + y, # Make flat list of bytes
      [bs for (bs, i) in self.know_values]))
    ints = [i for (bs, i) in self.know_values]

    self.assertEqual(list(le_to_int24(bytes)), ints)
    self.assertEqual(list(int24_to_le(ints)), bytes)

import pylab 
class TestBDFReader(unittest.TestCase):
  def setUp(self):
    self.bdf = BDFReader(open(os.path.join('data', 'sine-256Hz.bdf'), 'rb'))

  def test_read_all(self):
    b = self.bdf
    eeg = b.read_all()
    self.assertEqual(eeg.shape[0], 
      b.bdf.header['n_records'] * max(b.bdf.header['n_samples_per_record']))
    eeg_fft = np.abs(np.fft.rfft(eeg[:, :-1], axis=0))
    freqs = np.fft.fftfreq(eeg.shape[0], 1./256)
    print freqs[157]
    pylab.plot(freqs[1:900], eeg_fft[1:900, :])
    pylab.ylim([0, 5e5])
    pylab.show()
    np.testing.assert_equal(eeg_fft[1:,:].argmax(axis=0), np.ones(16) * 157)
