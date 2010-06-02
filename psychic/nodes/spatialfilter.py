import numpy as np
from numpy import linalg as la
from golem import DataSet
from golem.nodes import BaseNode

PLAIN, TRIAL, COV = range(3)

class BaseSpatialFilter(BaseNode):
  '''
  Handles the application of a spatial filter matrix W to different types
  of datasets.
  '''
  def __init__(self, ftype):
    BaseNode.__init__(self)
    self.W = None
    self.ftype = ftype

  def get_covs(self, d):
    '''
    Returns a list (or array) containing covariances, taking into account the
    type (flat, trial or cov) of the filter.
    '''
    if self.ftype == TRIAL:
      assert len(d.feat_shape) == 2
      return [np.cov(x, rowvar=False) for x in d.nd_xs]
    if self.ftype == COV:
      assert len(d.feat_shape) == 2 and d.feat_shape[0] == d.feat_shape[1]
      return d.nd_xs
    if self.ftype == PLAIN:
      assert len(d.feat_shape) == 1
      return [np.cov(d.xs, rowvar=False)]

  def sfilter(self, d):
    W = self.W
    if self.ftype == TRIAL:
      xs = np.array([np.dot(t, W) for t in d.nd_xs])
    elif self.ftype == COV:
      xs = np.array([reduce(np.dot, [W.T, t, W]) for t in d.nd_xs])
    elif self.ftype == PLAIN:
      xs = np.dot(d.xs, W)

    feat_shape = xs.shape[1:]
    xs = xs.reshape(xs.shape[0], -1)
    return DataSet(xs=xs, feat_shape=feat_shape, default=d)

  def apply_(self, d):
    return self.sfilter(d)

class CSP(BaseSpatialFilter):
  def __init__(self, m, ftype=TRIAL):
    BaseSpatialFilter.__init__(self, ftype)
    self.m = m

  def train_(self, d):
    assert d.nclasses == 2
    a = d.get_class(0)
    b = d.get_class(1)
    self.W = csp(self.get_covs(a), self.get_covs(b), self.m)

class CAR(BaseSpatialFilter):
  def __init__(self, n, ftype=PLAIN):
    BaseSpatialFilter.__init__(self, ftype)
    self.W = car(n)

def cov0(X):
  '''
  Calculate X^T X, a covariance estimate for zero-mean data without 
  normalizatoin.
  '''
  return np.dot(X.T, X)

def car(n):
  return np.eye(n) - np.ones((n, n)) / float(n)

def whitening(sigma):
  U, l, _ = la.svd(la.pinv(sigma))
  return np.dot(U, np.diag(l) ** .5)

def sym_whitening(sigma):
  U, l, _ = la.svd(la.pinv(sigma))
  return reduce(np.dot, [U, np.diag(l) ** .5, U.T])

def outer_n(n):
  '''Return a list with indices from both ends, i.e.: [0, 1, 2, -3, -2, -1]'''
  return np.roll(np.arange(n) - n/2, (n + 1) / 2)

def csp_base(sigma_a, sigma_b):
  '''Return CSP transformation matrix. No dimension reduction is performed.'''
  P = whitening(sigma_a + sigma_b)
  P_sigma_a = reduce(np.dot, [P.T, sigma_a, P])
  B, l, _ = la.svd(la.pinv(P_sigma_a))
  return np.dot(P, B)

def csp(sigma_a, sigma_b, m):
  '''
  Return a CSP transform for the covariance for class a and class b,
  with the m outer (~discriminating) spatial filters.
  '''
  return csp_base(sigma_a, sigma_b)[:, outer_n(m)]

def select_channels(n, keep_inds):
  '''
  Spatial filter to select channels keep_inds out of n channels. 
  Keep_inds can be both a list with indices, or an array of type bool.
  '''
  return np.eye(n)[:, keep_inds]

def deflate(sigma, noise_inds):
  '''
  Remove cross-correlation between noise channels and the rest. Based on [1].
  Note that *no channels are removed in the process*.

  [1] Alois Schloegl, Claudia Keinrath, Doris Zimmermann, Reinhold Scherer,
  Robert Leeb, and Gert Pfurtscheller. A fully automated correction method of
  EOG artifacts in EEG recordings. Clinical Neurophysiology, 118:98--104, 2007.
  '''
  n = sigma.shape[0]
  mask = np.zeros(n, np.bool)
  mask[noise_inds] = True

  # Find B, that predicts the EEG from EOG
  Cnn = sigma[mask][:, mask]
  Cny = sigma[mask][:, ~mask]
  B = np.dot(la.pinv(Cnn), Cny)
  B = np.hstack([B, np.zeros((B.shape[0], B.shape[0]))])

  # Construct final W
  W = np.eye(n) - np.dot(select_channels(n, noise_inds), B)
  return W[:, ~mask]

# ----------------------
import unittest
import matplotlib.pyplot as plt

class TestBaseSpatialFilter(unittest.TestCase):
  def setUp(self):
    # build dataset with artificial trials
    dtrial = DataSet(xs=np.random.rand(10, 32 * 128) + 100, 
      feat_shape=(128, 32), ys=np.zeros((10, 1)))
    
    # derive cov-based dataset
    covs = np.asarray([np.cov(t, rowvar=False) for t in dtrial.nd_xs])
    dcov = DataSet(xs=covs.reshape(covs.shape[0], -1), 
      feat_shape=covs.shape[1:], default=dtrial)

    # construct plain dataset (without trials) based on dtrail
    xs = np.vstack(dtrial.nd_xs)
    dplain = DataSet(xs=xs, ys=np.zeros((xs.shape[0], 1)))

    self.dplain = dplain
    self.dtrail = dtrial
    self.dcov = dcov

  def test_plain(self):
    d = self.dplain
    print d
    f = BaseSpatialFilter(ftype=PLAIN)
    f.W = np.random.randn(32, 4)
    np.testing.assert_equal(f.get_covs(d), [np.cov(d.xs, rowvar=False)])
    np.testing.assert_equal(f.sfilter(d).xs, np.dot(d.xs, f.W))

  def test_trial(self):
    dtrial = self.dtrail
    f = BaseSpatialFilter(ftype=TRIAL)
    f.W = np.random.randn(32, 4)

    # test that the covariances are correctly extracted
    covs = np.asarray([np.cov(t, rowvar=False) for t in dtrial.nd_xs])
    np.testing.assert_equal(f.get_covs(dtrial), covs)

    # verify that the mapping is applied correctly
    np.testing.assert_equal(f.sfilter(dtrial).nd_xs, 
      [np.dot(t, f.W) for t in dtrial.nd_xs])

  def test_cov(self):
    dtrial = self.dtrail
    dcov = self.dcov

    f = BaseSpatialFilter(ftype=COV)
    f.W = np.random.randn(32, 4)

    # test that the covariances are correctly extracted
    np.testing.assert_equal(f.get_covs(dcov), dcov.nd_xs)

    # verify that the mapping is applied correctly
    target = np.array([np.cov(np.dot(t, f.W), rowvar=False) for t in 
      dtrial.nd_xs])
    np.testing.assert_almost_equal(f.sfilter(dcov).nd_xs, target)
      


class TestSpatialFilters(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)

  def test_select_channels(self):
    xs = np.random.rand(40, 10)
    for keep in [[0, 1, -3, 2], (np.arange(10) % 2 == 0).astype(bool)]:
      np.testing.assert_equal(np.dot(xs, select_channels(xs.shape[1], keep)), 
        xs[:, keep])

  def test_deflate(self):
    xs = np.dot(np.random.randn(1000, 4), np.random.rand(4, 4))
    xs = np.hstack([xs, np.random.rand(xs.shape[0], 2) * 20])
    xs = xs - np.mean(xs, axis=0)

    A = np.eye(6)
    A[-2:,:-2] = np.random.rand(2, 4)
    xs_mix = np.dot(xs, A)

    W = deflate(np.cov(xs_mix, rowvar=False), [-2, -1])
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
    W = whitening(np.cov(xs, rowvar=False))
    xs2 = np.dot(xs, W)
    self.assertEqual(xs2.shape, xs.shape)
    np.testing.assert_almost_equal(np.cov(xs2, rowvar=False), np.eye(5))

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
    xa = np.random.randn(100, 8) * np.array([1, 1, 1, 3, 1, 1, 1, 1])
    xb = np.random.randn(100, 8) * np.array([1, 1, 1, .1, 1, 1, 1, 1])

    sig_a = np.cov(xa, rowvar=False)
    sig_b = np.cov(xb, rowvar=False)

    W = csp_base(sig_a, sig_b)
    self.assertEqual(W.shape, (8, 8))
    D1 = np.cov(np.dot(xa, W), rowvar=False)
    D2 = np.cov(np.dot(xb, W), rowvar=False)

    self.assert_(np.allclose(D1 + D2, np.eye(D1.shape[0])), 
      'Joint covariance is not the identity matrix.')
    self.assert_(np.allclose(np.diag(np.diag(D1)), D1),
      'Class covariance is not diagonal.')
    self.assert_(np.allclose(np.diag(D1), np.sort(np.diag(D1))),
      'Class variance is not ascending.')


if __name__ == '__main__':
  unittest.main()
