import unittest, os
import numpy as np
from golem import DataSet
from .. import utils

class TestEventDetection(unittest.TestCase):
  def setUp(self):
    pass

  def do_test(self, status, events, indices):
    e, i = utils.status_to_events(status)
    np.testing.assert_equal(e, events)
    np.testing.assert_equal(i, indices)

  def test_single_event(self):
    self.do_test([0, 0, 0, 1, 1, 0, 0, 0], [1], [3])

  def test_functionality(self):
    self.do_test([2, 0, 1, 1, 3, 0, 4], [2, 1, 3, 4], [0, 2, 4, 6])

  def test_degenerate(self):
    self.do_test([0], [], [])


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

      
class TestSlice(unittest.TestCase):
  def setUp(self):
    self.frames = np.arange(40).reshape(-1, 2)

  def test_slice(self):
    slices = utils.slice(self.frames, [2, 16], offsets=[-2, 4])
    self.assertEqual(slices.shape, (2, 6, 2))
    np.testing.assert_equal(slices[0, :, :], np.arange(0, 12).reshape(-1, 2))
    np.testing.assert_equal(slices[1, :, :], np.arange(28, 40).reshape(-1, 2))

  def test_outside(self):
    np.testing.assert_equal(
      utils.slice(self.frames, [1, 3, 19], offsets=[-2, 4]),
      utils.slice(self.frames, [3], offsets=[-2, 4]))

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
    
    # test sample rate
    self.assertEqual(d.extra, {'sample_rate' : 256.})

class TestResampleRec(unittest.TestCase):
  def setUp(self):
    xs = np.arange(100).reshape(-1, 2)
    ys = np.zeros((50, 1))
    ys[::4] = 2
    self.d = DataSet(xs=xs, ys=ys, extra={'sample_rate': 20})

  def test_resample(self):
    d = self.d
    d2 = utils.resample_rec(d, 10) # factor .5
    self.assertEqual(d2.ninstances, d.ninstances / 2)
    self.assertEqual(d2.nfeatures, d.nfeatures)
    self.assertEqual(d2.feat_lab, d.feat_lab)
    self.assertEqual(d2.cl_lab, d.cl_lab)
    self.assertEqual(d2.feat_shape, d.feat_shape)
    np.testing.assert_equal(d2.ys[::2], np.ones((13, 1)) * 2)
    self.assertEqual(d2.extra, {'sample_rate': 10})

  def test_overlapping_markers(self):
    d = self.d

    # test overlapping markers
    self.assertRaises(AssertionError, utils.resample_rec, d, 4)

    # test too-tightly packed markers
    self.assertRaises(AssertionError, utils.resample_rec, d, 5)
