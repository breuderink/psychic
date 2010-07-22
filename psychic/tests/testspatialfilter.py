import unittest
import numpy as np
from numpy import linalg as la
from golem import DataSet
from golem.nodes import BaseNode

from ..nodes.spatialfilter import *

class TestBaseSpatialFilter(unittest.TestCase):
  def setUp(self):
    # build dataset with artificial trials
    dtrial = DataSet(xs=np.random.randn(10, 32 * 128),
      feat_shape=(128, 32), ys=np.zeros((10, 1)))
    
    # derive cov-based dataset
    covs = np.asarray([np.cov(t, rowvar=False) for t in dtrial.nd_xs])
    dcov = DataSet(xs=covs.reshape(covs.shape[0], -1), 
      feat_shape=covs.shape[1:], default=dtrial)

    # construct plain dataset (without trials) based on dtrial
    xs = np.vstack(dtrial.nd_xs)
    dplain = DataSet(xs=xs, ys=np.zeros((xs.shape[0], 1)))

    self.dplain = dplain
    self.dtrial = dtrial
    self.dcov = dcov

  def test_plain(self):
    d = self.dplain
    f = BaseSpatialFilter(ftype=PLAIN)
    f.W = np.random.randn(32, 4)

    self.assertEqual(f.get_nchannels(d), 32)
    np.testing.assert_equal(f.get_cov(d), cov0(d.xs))
    np.testing.assert_equal(f.sfilter(d).xs, np.dot(d.xs, f.W))

  def test_trial(self):
    dtrial = self.dtrial
    f = BaseSpatialFilter(ftype=TRIAL)
    f.W = np.random.randn(32, 4)

    # test extraction of number of channels
    self.assertEqual(f.get_nchannels(dtrial), 32)

    # Test that the covariances are correctly extracted. The devision by n-1
    # causes some small differences.
    np.testing.assert_almost_equal(f.get_cov(dtrial), 
      cov0(np.vstack(dtrial.nd_xs)), decimal=2)

    # verify that the mapping is applied correctly
    np.testing.assert_equal(f.sfilter(dtrial).nd_xs, 
      [np.dot(t, f.W) for t in dtrial.nd_xs])

  def test_cov(self):
    dtrial = self.dtrial
    dcov = self.dcov

    f = BaseSpatialFilter(ftype=COV)
    f.W = np.random.randn(32, 4)

    # test extraction of number of channels
    self.assertEqual(f.get_nchannels(dcov), 32)

    # test that the covariances are correctly extracted
    np.testing.assert_equal(f.get_cov(dcov), np.mean(dcov.nd_xs, axis=0))

    # verify that the mapping is applied correctly
    target = np.array([np.cov(np.dot(t, f.W), rowvar=False) for t in 
      dtrial.nd_xs])
    np.testing.assert_almost_equal(f.sfilter(dcov).nd_xs, target)
      

class TestSpatialFilters(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)

  def test_cov0(self):
    xs = np.dot(np.random.rand(100, 4), np.random.rand(4, 4))
    xs = xs - np.mean(xs, axis=0)
    np.testing.assert_almost_equal(cov0(xs), 
      np.cov(xs, rowvar=False))

  def test_select_channels(self):
    xs = np.random.rand(40, 10)
    for keep in [[0, 1, -3, 2], (np.arange(10) % 2 == 0).astype(bool)]:
      np.testing.assert_equal(np.dot(xs, select_channels(xs.shape[1], keep)), 
        xs[:, keep])

  def test_deflate(self):
    # make centered xs, with 2 big amplitude channels at the end.
    xs = np.dot(np.random.randn(1000, 4), np.eye(4))
    xs = np.hstack([xs, np.random.randn(xs.shape[0], 2) * 20])

    # spread some influence of the big amplitude channels.
    A = np.eye(6)
    A[-2:,:-2] = np.random.rand(2, 4)
    xs_mix = np.dot(xs, A)
    xs_mix = xs_mix - np.mean(xs_mix, axis=0)

    # Verify that it undoes the mixing. I suspect that the poor numerical 
    # precision is the result of random correlations in xs.
    sig = cov0(xs_mix)
    sig_S = cov0(xs)

    W = deflate(sig, [False, False, False, False, True, True])

    np.testing.assert_almost_equal(reduce(np.dot, [W.T, sig, W]), 
      cov0(xs[:, :4]), decimal=2)
    np.testing.assert_almost_equal(np.dot(A, W), np.eye(6)[:, :-2], decimal=2)

  def test_car(self):
    xs = np.random.rand(10, 4)
    W = car(xs.shape[1])
    self.assert_(np.allclose(np.dot(xs, W), xs - 
      np.mean(xs, axis=1).reshape(-1,1)))

  def test_outer_n(self):
    np.testing.assert_equal(outer_n(1), [0])
    np.testing.assert_equal(outer_n(2), [0, -1])
    np.testing.assert_equal(outer_n(3), [0, 1, -1])
    np.testing.assert_equal(outer_n(6), [0, 1, 2, -3, -2, -1])

  def test_whitening(self):
    xs = np.random.randn(100, 5)
    W = whitening(cov0(xs))
    xs2 = np.dot(xs, W)
    self.assertEqual(xs2.shape, xs.shape)
    np.testing.assert_almost_equal(cov0(xs2), np.eye(5))

  def test_whitening_lowrank(self):
    xs = np.dot(np.random.randn(100, 3), np.random.rand(3, 5))
    W = whitening(cov0(xs))
    xs2 = np.dot(xs, W)
    np.testing.assert_almost_equal(cov0(xs2), np.eye(3))

  def test_sym_whitening(self):
    xs = np.random.randn(100, 5)
    W = sym_whitening(np.cov(xs, rowvar=False))
    xs2 = np.dot(xs, W)

    # test whitening property
    self.assertEqual(xs2.shape, xs.shape)
    np.testing.assert_almost_equal(np.cov(xs2, rowvar=False), np.eye(5))

    # test symmetry
    np.testing.assert_almost_equal(W, W.T)

  def test_csp(self):
    xa = np.random.randn(100, 4) * np.array([1, 1, 1, 3])
    xb = np.random.randn(100, 4) * np.array([1, 1, 1, .1])

    # create low-rank data
    A = np.random.rand(4, 8)
    xa = np.dot(xa, A)
    xb = np.dot(xb, A)

    sig_a = cov0(xa)
    sig_b = cov0(xb)

    for m in range(2, 6):
      W = csp(sig_a, sig_b, m)
      self.assertEqual(W.shape, (8, min(m, 4)))
      D1 = cov0(np.dot(xa, W))
      D2 = cov0(np.dot(xb, W))

      np.testing.assert_almost_equal(D1 + D2, np.eye(W.shape[1]), 
        err_msg='Joint covariance is not the identity matrix.')
      np.testing.assert_almost_equal(np.diag(np.diag(D1)), D1,
        err_msg='Class covariance is not diagonal.')
      np.testing.assert_almost_equal(np.diag(D1), np.sort(np.diag(D1)),
        err_msg='Class variance is not ascending.')
