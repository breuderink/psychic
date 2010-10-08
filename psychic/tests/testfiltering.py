import unittest, os, operator
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from golem import DataSet
from ..plots import plot_timeseries
from .. import filtering
from ..nodes import OnlineFilter, Winsorize

class TestResample(unittest.TestCase):
  def setUp(self):
    xs = np.arange(1000).reshape(-1, 2)
    ys = np.zeros((500, 1))
    ys[::4] = 2
    self.d = DataSet(xs=xs, ys=ys)

  def test_resample(self): 
    d = self.d
    d2 = filtering.resample_rec(d, .5)
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
    self.assertRaises(AssertionError, filtering.decimate_rec, d, 5)


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

    d2 = filtering.decimate_rec(d, 2)
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
    d2 = filtering.decimate_rec(d, 2)
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
    self.assertRaises(AssertionError, filtering.decimate_rec, d, 4)

class TestFilter(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)
    self.d = DataSet(xs=np.random.rand(400, 4), ys=np.zeros((400, 1)))

  def test_nop(self):
    b, a = np.array([0, 1]), np.array([1])
    self.assertEqual(filtering.filtfilt_rec(self.d, (b, a)), self.d)

  def test_lp(self):
    b, a = signal.iirfilter(4, [.1], btype='low')
    df = filtering.filtfilt_rec(self.d, (b, a))
    spec = np.abs(np.fft.rfft(df.xs, axis=0))

    # verify that there is more power in the lowest 10%
    pass_p = np.mean(spec[:self.d.ninstances/10], axis=0)
    stop_p = np.mean(spec[self.d.ninstances/10:], axis=0)
    self.assert_(((pass_p/stop_p) > 20).all())

  def test_hp(self):
    b, a = signal.iirfilter(6, [.9], btype='high')
    df = filtering.filtfilt_rec(self.d, (b, a))
    # only test for zero mean
    np.testing.assert_almost_equal(np.mean(df.xs, axis=0), np.zeros(4), 3)

class TestOnlineFilter(unittest.TestCase):
  def test_online_filter(self):
    N = 200
    WIN = 50

    d = DataSet(xs=np.random.rand(N, 3) + 100, ys=np.zeros((N, 1)))
    d0, stream = d[:10], d[10:]

    def filt_design_f(sr):
      return signal.iirfilter(4, [.01, .2])

    of = OnlineFilter(filt_design_f)
    of.train(d0) # get sampling rate for filter design

    store = []
    tail = stream
    while len(tail) > 0:
     head, tail = tail[:WIN], tail[WIN:]
     store.append(of.apply(head))
    filt_d = reduce(operator.add, store)

    b, a = of.filter
    np.testing.assert_equal(filt_d.xs,  signal.lfilter(b, a, stream.xs, axis=0))

class TestWinsorizing(unittest.TestCase):
  def setUp(self):
    xs = np.random.rand(100, 5) + np.arange(5)
    xs[10, :] = 10;
    xs[11, :] = -10
    self.d = DataSet(xs=xs, ys=np.ones((100, 1)))

  def test_nop(self):
    d = self.d
    nop = Winsorize([0, 1]).train_apply(d, d)
    self.assertEqual(nop, d)

  def test_minimal(self):
    d = self.d
    wd = Winsorize([.01, .99]).train_apply(d, d)
    self.assert_(np.all((wd.xs == d.xs)[:10]))
    self.assert_(np.all((wd.xs != d.xs)[10:12]))
    self.assert_(np.all((wd.xs == d.xs)[12:]))

def ewma_ref(x, alpha, v0=0):
  x = np.atleast_1d(x).flatten()
  result = np.zeros(x.size + 1)
  result[0] = v0

  for i in range(1, x.size + 1):
    result[i] = alpha * result[i-1] + (1.-alpha) * x[i-1]
  return result[1:]

class TestMAs(unittest.TestCase):
  def setUp(self):
    np.random.seed(3)
    self.s = np.cumsum(np.random.randn(1000)) + np.random.rand(1000) + 40
    self.s[400] = 100

  def test_ma(self):
    for n in [2, 10, 60]:
      np.testing.assert_almost_equal(
        signal.lfilter(np.ones(n), float(n), self.s), filtering.ma(self.s, n))

  def test_emwa(self):
    s = self.s
    for alpha in [.93, .95, .97, .999]:
      np.testing.assert_almost_equal(
        filtering.ewma(s, alpha), ewma_ref(s, alpha))

    for alpha in [.93, .95, .97, .999]:
      for v0 in [0, .01, 4]:
        np.testing.assert_almost_equal(
          filtering.ewma(s, alpha, v0), ewma_ref(s, alpha, v0))
