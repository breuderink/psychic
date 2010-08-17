import unittest, os.path, logging
import numpy as np
from golem import DataSet, helpers
from .. import utils
from .. import markers

import matplotlib.pyplot as plt

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
  # The key principle we are using for validation is:
  # np.mean(np.abs(np.fft.fft(s)) ** 2) - np.sum(s**2) < 1e10
  # See http://en.wikipedia.org/wiki/Parseval%27s_theorem#Applications
  def setUp(self):
    self.FS = FS = 512
    self.N = N = FS * 20
    self.NFFT = NFFT = 256
    self.alpha = np.sin(np.linspace(0, 12 * 2 * np.pi * N/FS, N))
    self.beta = np.sin(np.linspace(0, 30 * 2 * np.pi * N/FS, N))

    self.alpha_beta = self.alpha + self.beta    
    self.beta_spike = self.beta.copy()
    self.beta_spike[self.N/2] = 100

  def test_stepsize(self):
    FS, N, NFFT = self.FS, self.N, self.NFFT
    s = self.alpha_beta
    energy_s = np.sum(s**2)
    for step in [NFFT, NFFT/2, NFFT/4, NFFT/8, 1]:
      spec = utils.spectrogram(s, NFFT, step)
      self.assertEqual(spec.shape[0], 1 + (s.size - NFFT) / step)
      self.assertEqual(spec.shape[1], NFFT/2 + 1)
      np.testing.assert_approx_equal(np.sum(spec), energy_s, 3)

  def test_NFFT(self):
    FS, N, NFFT = self.FS, self.N, self.NFFT
    s = self.alpha_beta
    energy_s = np.sum(s**2)
    for NFFT in [4, 8, 9, 31, 32, 33, 128, 512]:
      spec = utils.spectrogram(s, NFFT, NFFT)
      self.assertEqual(spec.shape[0], s.size / NFFT)
      self.assertEqual(spec.shape[1], NFFT/2 + 1)
      np.testing.assert_approx_equal(np.sum(spec), energy_s, 3)

  def test_temp_power(self):
    FS, N, NFFT = self.FS, self.N, self.NFFT
    sig = self.beta_spike

    step = NFFT / 2
    spec = utils.spectrogram(sig, NFFT, step)

    nspec, nsig = spec.shape[0], sig.size
    cuts = np.linspace(0, 1, 8)
    for (start, end) in zip(cuts[:-1], (cuts[1:])):
      energy_spec = np.sum(spec[nspec * start:nspec * end])
      energy_sig = np.sum(sig[nsig * start:nsig * end] ** 2)
      np.testing.assert_approx_equal(energy_spec, energy_sig, 1)

  def test_freq_power(self):
    FS, N, NFFT = self.FS, self.N, self.NFFT
    sig = self.alpha_beta
    spec = utils.spectrogram(sig, NFFT, NFFT/2)

    freqs = np.fft.fftfreq(NFFT, 1./FS)[:spec.shape[1]]
    alpha_mask = freqs < 20
    beta_mask = freqs >= 20

    np.testing.assert_approx_equal(np.sum(spec), np.sum(sig ** 2), 3)
    np.testing.assert_approx_equal(np.sum(spec[:,alpha_mask]), 
      np.sum(self.alpha ** 2), 3)
    np.testing.assert_approx_equal(np.sum(spec[:,beta_mask]), 
      np.sum(self.beta ** 2), 3)

  def test_plot(self):
    FS, N, NFFT = self.FS, self.N, self.NFFT
    plt.clf()
    plt.imshow(10 * np.log10(utils.spectrogram(
      self.beta_spike, NFFT, NFFT/2).T), interpolation='nearest',
      origin='lower')
    plt.savefig(os.path.join('out', 'spec.eps'))
  
  def test_wave_spike(self):
    FS, N, NFFT = self.FS, self.N, self.NFFT
    spec = utils.spectrogram(self.beta_spike, NFFT, NFFT/2)
    # no negative values
    self.assert_((spec > 0).all())

    # verify that the spike is centered in time
    self.assertEqual(spec.shape, (79, NFFT/2 + 1))
    self.assertEqual(np.argmax(np.mean(spec, axis=1)), spec.shape[0]/2)

    # verify that the peak frequency ~ 30Hz
    freqs = np.fft.fftfreq(NFFT, 1./FS)
    beta_i = np.argmin(np.abs(freqs - 30))
    self.assertEqual(np.argmax(np.mean(spec, axis=0)), beta_i)
      

class TestBDF(unittest.TestCase):
  def test_load(self):
    d = utils.load_bdf(os.path.join('data', 'sine-256Hz.bdf'))

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
    ids = np.hstack([np.arange(20).reshape(-1, 1), np.ones((20, 1), int)])
    self.d = DataSet(xs=xs, ys=ys, ids=ids)

  def test_windows(self):
    xs = np.random.rand(20, 5)
    ys = np.zeros((20, 1))
    ys[[2, 16]] = 1
    ys[[4, 12]] = 2

    d2 = utils.slice(DataSet(xs=xs, ys=ys), {1:'b', 2:'a'}, offsets=[-2, 4])

    np.testing.assert_equal(d2.ids.flatten(), [2, 4, 12, 16])

    win_base = np.arange(12).reshape(-1, 2)
    for i, off in enumerate([2, 4, 12, 16]):
      np.testing.assert_equal(d2.nd_xs[i], xs[off-2:off+4])

  def test_labels(self):
    xs = np.random.rand(20, 5)
    ys = np.zeros((20, 1))
    ys[[2, 16]] = 1
    ys[[4, 12]] = 2
    ys[[8]] = 3
    ys[[9]] = 4
    d2 = utils.slice(DataSet(xs=xs, ys=ys), {1:'b', 2:'a', 3:'c', 4:'c'}, 
      offsets=[0, 2])
    self.assertEqual(d2.cl_lab, ['a', 'b', 'c'])
    np.testing.assert_equal(d2.get_class(0).ids.flatten(), [4, 12])
    np.testing.assert_equal(d2.get_class(1).ids.flatten(), [2, 16])
    np.testing.assert_equal(d2.get_class(2).ids.flatten(), [8, 9])

  def test_few_trials(self):
    xs = np.random.rand(20, 5)
    ys = np.zeros((20, 1))

    mdict = {1:'b', 2:'a'}
  
    ds = utils.slice(DataSet(xs=xs, ys=ys.copy()), mdict, [0, 2])
    self.assertEqual(ds.ninstances_per_class, [0, 0])
    
    ys[5] = 1
    ds = utils.slice(DataSet(xs=xs, ys=ys.copy()), mdict, [0, 2])
    self.assertEqual(ds.ninstances_per_class, [0, 1])

    ys[5] = 2
    ds = utils.slice(DataSet(xs=xs, ys=ys.copy()), mdict, [0, 2])
    self.assertEqual(ds.ninstances_per_class, [1, 0])

    ys[6] = 1
    ds = utils.slice(DataSet(xs=xs, ys=ys.copy()), mdict, [0, 2])
    self.assertEqual(ds.ninstances_per_class, [1, 1])

  def test_bounds(self):
    xs = np.random.rand(20, 5)
    ys = np.zeros((20, 1))
    ys[[3, 20-3]] = 1
    ys[[4, 20-4]] = 2
    logging.getLogger('psychic.utils.slice').setLevel(logging.ERROR)
    ds = utils.slice(DataSet(xs=xs, ys=ys), {1:'bad', 2:'good'}, [-4, 4])
    self.assertEqual(ds.ninstances, 2)
    np.testing.assert_equal(ds.ids.flatten(), [4, 20-4])
    logging.getLogger('psychic.utils.slice').setLevel(logging.WARNING)

  def test_feat_labs(self):
    xs = np.random.rand(20, 2)
    ys = np.zeros((20, 1))
    ys[10] = 1
    ids = np.arange(20).reshape(-1, 1) / 10.
    d = DataSet(xs=xs, ys=ys, ids=ids, feat_lab=['a', 'b'])
    ds = utils.slice(d, {1:'hit'}, [-2, 2])

    self.assertEqual(ds.feat_nd_lab[0], ['%.3f' % (i / 10.) for i in 
      range(-2, 2)])
    self.assertEqual(ds.feat_nd_lab[1], d.feat_lab)

  def test_nd_ids(self):
    xs = np.random.rand(20, 2)
    ys = np.zeros((20, 1))
    ys[[10, 12]] = 1
    ids = np.hstack([np.arange(20).reshape(-1, 1), np.ones((20, 1))])
    ds = utils.slice(DataSet(xs=xs, ys=ys, ids=ids), {1:'hit'}, [-2, 2])
    np.testing.assert_equal(ds.ids, [[10, 1], [12, 1]])


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

class TestGetSamplerate(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)

  def test_get_samplerate(self):
    d = DataSet(xs=np.random.rand(100, 10), ys=np.zeros((100, 1)))
    for f in [2, 10, 2048]:
      df = DataSet(ids = np.arange(100).reshape(-1, 1)/float(f), default=d)
      self.assertEqual(utils.get_samplerate(df), f)

  def test_noise(self):
    d = DataSet(xs=np.random.rand(100, 10), ys=np.zeros((100, 1)), 
      ids=np.arange(100).reshape(-1, 1) / 10. + 0.01 * np.random.rand(100, 1))
    self.assertEqual(utils.get_samplerate(d), 10)


class TestBitrate(unittest.TestCase):
  def test_wolpaw(self):
    self.assertAlmostEqual(utils.wolpaw_bitr(2, 1/2.), 0)
    self.assertAlmostEqual(utils.wolpaw_bitr(2, 1), 1)
    self.assertAlmostEqual(utils.wolpaw_bitr(2, 0), 1)
    self.assertAlmostEqual(utils.wolpaw_bitr(3, 1/3.), 0)
    self.assertAlmostEqual(utils.wolpaw_bitr(4, 1/4.), 0)
    self.assertAlmostEqual(utils.wolpaw_bitr(4, 1), 2)
