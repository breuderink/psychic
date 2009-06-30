import unittest, os, logging
import numpy as np
from golem import DataSet, helpers
from .. import utils

class TestSlidingWindow(unittest.TestCase):
  def test_indices(self):
    windows = utils.sliding_window_indices(5, 2, 10)
    self.assertEqual(windows.shape, (3, 5))
    np.testing.assert_equal(windows[:, 0], [0, 2, 4])
    np.testing.assert_equal(windows[0, :], range(5))

  def test_functionality_1D(self):
    signal = np.arange(10)
    windows = utils.sliding_window(signal, 5, 2)
    self.assertEqual(windows.shape, (3, 5))
    np.testing.assert_equal(windows[:, 0], [0, 2, 4])
    np.testing.assert_equal(windows[0, :], range(5))

  def test_winf(self):
    signal = np.arange(10)
    winf = np.arange(5)
    windows = utils.sliding_window(signal, 5, 2, winf)
    self.assertEqual(windows.shape, (3, 5))
    np.testing.assert_equal(windows[:, 0], [0, 0, 0])
    np.testing.assert_equal(windows[0, :], np.arange(5) ** 2)

    self.assertEqual(utils.sliding_window(signal, 5, 2, np.hanning(5)).dtype, 
      np.hanning(5).dtype)

  def test_input_errors(self):
    signal = np.arange(10)

    # invalid window function
    self.assertRaises(ValueError, utils.sliding_window, 
      signal, 5, 2, np.arange(6))
    self.assertRaises(ValueError, utils.sliding_window, 
      signal, 5, 2, np.arange(2))

    # invalid shape
    self.assertRaises(ValueError, 
      utils.sliding_window, signal.reshape(5, 2), 5, 2, np.arange(6))


class TestSTFT(unittest.TestCase):
  def test_stft(self):
    # assume the windowing is already tested...
    # compare FFT of hanning-windowed signal with STFT
    signal = np.random.rand(20)
    windows = utils.sliding_window(signal, 5, 2, np.hanning(5))
    np.testing.assert_equal(np.fft.rfft(windows, axis=1), 
      utils.stft(signal, 5, 2))


class TestSpectrogram(unittest.TestCase):
  def test_wave_spike(self):
    beta_spike = np.sin(np.linspace(0, 30 * 2 * np.pi, 512))
    beta_spike[256] = 100
    spec = utils.spectrogram(beta_spike, 64, 32)

    # no negative values
    self.assert_((spec > 0).all())

    # verify that the spike is centered in time
    self.assertEqual(spec.shape[0], 15)
    self.assertEqual(np.argmax(np.mean(spec, axis=1)), 7)

    # verify that the peak frequency ~ 30Hz
    freqs = np.fft.fftfreq(64, 1./512)
    beta_i = np.argmin(np.abs(freqs - 30))
    self.assertEqual(np.argmax(np.mean(spec, axis=0)), beta_i)


class TestPopcorn(unittest.TestCase):
  def setUp(self):
    self.signals = np.arange(4 * 6 * 8 * 1 * 2 * 3).reshape(4, 6, 8, -1)

  def test_noexpansion(self):
    times2 = lambda signal: signal * 2
    for axis in range(self.signals.ndim):
      signals2 = utils.popcorn(times2, axis, self.signals)
      self.assertEqual(self.signals.shape, signals2.shape)
      np.testing.assert_equal(self.signals * 2, signals2)

  def test_adddim(self):
    add_dims = lambda signal, n: np.array([signal] * n)
    sss = self.signals.shape
    for nd in range(1, 4):
      for axis in range(self.signals.ndim):
        signals2 = utils.popcorn(add_dims, axis, self.signals, nd)
        self.assertEqual(sss[:axis] + (nd,) + sss[axis:], signals2.shape)
        # remove new axis and test for equality
        np.testing.assert_equal(self.signals, np.rollaxis(signals2, axis, 0)[0])

  def test_reshape(self):
    reshape_1d = lambda signal: signal.reshape(2, -1)
    sss = self.signals.shape
    for axis in range(self.signals.ndim):
      signals2 = utils.popcorn(reshape_1d, axis, self.signals)
      self.assertEqual(sss[:axis], signals2.shape[:axis])
      self.assertEqual(sss[axis], np.prod(signals2.shape[axis:axis+2]))
      self.assertEqual(sss[axis+1:], signals2.shape[axis+2:])
      np.testing.assert_equal(self.signals, signals2.reshape(sss))

      
class TestBDF(unittest.TestCase):
  def test_load(self):
    d = utils.bdf_dataset(os.path.join('data', 'sine-256Hz.bdf'))

    # test labels
    targets = ['A%d' % (i + 1) for i in range(16)]
    self.assertEqual(d.feat_lab, targets)
    self.assertEqual(d.cl_lab, ['status'])

    # test ids ~ time
    self.assertAlmostEqual(d.ids[256 + 1], 1, 2)

    # test dims
    self.assertEqual(d.nfeatures, 16)
    self.assertEqual(d.ninstances, 60 * 256)
    
    self.assertEqual(d.extra, {})


class TestSlice(unittest.TestCase):
  def setUp(self):
    xs = np.arange(40).reshape(-1, 2)
    ys = np.zeros((20, 1))
    ys[[0, 2, 16], 0] = 1
    ys[[4, 12, 19], 0] = 2
    ids = np.hstack([np.arange(20).reshape(-1, 1), np.ones((20, 1))])
    self.d = DataSet(xs=xs, ys=ys, ids=ids)

  def test_slice(self):
    logging.getLogger('psychic.utils.slice').setLevel(logging.ERROR)
    d2 = utils.slice(self.d, dict(b=1, a=2), offsets=[-2, 4])
    logging.getLogger('psychic.utils.slice').setLevel(logging.WARNING)
    self.assertEqual(d2.feat_shape, (6, 2))
    self.assertEqual(d2.ninstances, 4)

    np.testing.assert_equal(d2.xs[0], np.arange(12))
    np.testing.assert_equal(d2.xs[1], np.arange(12) + (4 - 2) * 2)
    np.testing.assert_equal(d2.xs[2], np.arange(12) + (12 - 2) * 2)
    np.testing.assert_equal(d2.xs[3], np.arange(12) + (16 - 2) * 2)

    np.testing.assert_equal(d2.ys, helpers.to_one_of_n([0, 1, 1, 0]))
    
    np.testing.assert_equal(d2.ids, [[0, 1], [2, 1], [10, 1], [14, 1]])

    self.assertEqual(d2.cl_lab, ['b', 'a'])
    self.assertEqual(d2.feat_lab, None)
    self.assertEqual(d2.feat_nd_lab, 
      [['-2.00', '-1.00', '0.00', '1.00', '2.00', '3.00'], ['f0', 'f1']])

    self.assertEqual(d2.feat_nd_lab[0], 
      ['%.2f' % i for i in d2.extra['event_time']])

class TestFindSegments(unittest.TestCase):
  def test_naive(self):
    self.assertEqual(utils.find_segments(
      [1, 3, 3, 2, 4, 4, 1, 3, 3, 2], np.arange(10)-2, 1, 2),
      [(-2, 1), (4, 7)])
      
  def test_overlapping(self):
    self.assertEqual(utils.find_segments([3, 1, 3, 4, 1, 4], range(6), 3, 4),
      [(2, 3), (0, 5)])

  def test_malformed(self):
    self.assertRaises(AssertionError, utils.find_segments, [4, 3, 4], 
      range(3), 3, 4)

  def test_openended(self):
    logging.getLogger('psychic.utils.find_segments').setLevel(logging.ERROR)
    self.assertEqual(utils.find_segments([3, 1], range(2), 3, 4), [])
    logging.getLogger('psychic.utils.find_segments').setLevel(logging.WARNING)

class TestCutSegments(unittest.TestCase):
  def setUp(self):
    ys = np.array([0, 0, 3, 0, 7, 4, 0, 0, 1, 0, 2, 1, 8, 0, 2]).reshape(-1, 1)
    xs = np.arange(ys.size).reshape(-1, 1)
    self.d = DataSet(xs, ys)

  def test_cut_segments(self):
    ds = utils.cut_segments(self.d, [(1, 2), (3, 4)])
    np.testing.assert_equal(ds[0].ids.flatten(), [2, 3, 4])
    np.testing.assert_equal(ds[1].ids.flatten(), [8, 9])
    np.testing.assert_equal(ds[2].ids.flatten(), [11, 12, 13])

  def test_cut_segments(self):
    ds = utils.cut_segments(self.d, [(1, 2), (3, 4)], offsets=[1, 1])
    np.testing.assert_equal(ds[0].ids.flatten(), [3, 4, 5])
    np.testing.assert_equal(ds[1].ids.flatten(), [9, 10])
    np.testing.assert_equal(ds[2].ids.flatten(), [12, 13, 14])
