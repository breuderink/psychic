import numpy as np
from numpy import linalg as la

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
  return np.roll(np.arange(n) - n/2, (n+1)/2)

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

import unittest
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
