import unittest
from ..utils import stft, get_samplerate
from ..nodes.nonstat import *
from golem import DataSet
from scipy import signal
import matplotlib.pyplot as plt

class TestSlowSphere(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)
    
  def test_slowsphere(self):
    '''
    When the input is stationary, the result should be similar to the
    symmetrical whitening transform but no low-frequency changes should
    exist.
    '''
    # init
    S = np.random.randn(1e4, 4)
    A = np.where(np.eye(4), 1, np.random.rand(4, 4))
    xs = np.dot(S, A)
    xs = xs-np.mean(xs, axis=0)

    # perform slow sphering
    xs2 = slow_sphere(xs, signal.iirfilter(2, .1, btype='low'), 10)

    # test slowness property
    wstep = 20
    sigs = np.asarray([cov0(xs2[i:i+wstep]) for i in 
      range(0, xs2.shape[0], wstep)])

    spec = np.log(np.abs(np.apply_along_axis(
      lambda x: np.mean(np.abs(stft(x, 256, 256)), axis=0), 0, sigs)))

    self.assert_(np.mean(spec[1:10]) < np.mean(spec[10:]))

    # test whiteness property 
    np.testing.assert_almost_equal(
      np.cov(xs2, rowvar=False), np.eye(4), decimal=1)
  
    # test that whitening preserves channel mapping
    np.testing.assert_equal(
      np.argmax(np.corrcoef(S.T, xs2.T)[:4, 4:], axis=1),
      np.arange(4))

  def test_nodes_filter(self):
    d = DataSet(xs=np.zeros((1000, 10)), ys=np.ones((1000, 1)),
      ids=np.arange(1000).reshape(-1, 1) * (1./128))
    self.assertEqual(get_samplerate(d), 128)

    for isi in [5, 10]:
      for reest in [.1, 1]:
        s = SlowSphering(isi=isi, reest=reest)
        s.train(d)

        nyq = (1./reest) / 2.
        f = (1./isi) / nyq

        w, h = signal.freqz(*s.fil)
        h = np.abs(h)
        filt_cutoff = np.argmin(np.diff(h)) / float(h.size)
        self.assertAlmostEqual(filt_cutoff, f, places=1)
