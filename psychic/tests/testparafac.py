import operator, logging, unittest
import numpy as np

from ..parafac import *

class TestParafac(unittest.TestCase): 
  def setUp(self):
    self.a = np.random.randn(3, 4)
    self.b = np.random.randn(3, 5)
    self.c = np.random.randn(3, 6)
    np.random.seed(0)

  def test_ribs(self):
    a, b, c = self.a, self.b, self.c
    ra, rb, rc = ribs([a, b, c])

    # test shape of ribs
    self.assertEqual(ra.shape, (3, 4, 1, 1))
    self.assertEqual(rb.shape, (3, 1, 5, 1))
    self.assertEqual(rc.shape, (3, 1, 1, 6))
    
    # test that the data is not modified
    for pre, post in zip([a, b, c], [ra, rb, rc]):
      np.testing.assert_equal(np.squeeze(post), pre)

  def test_normalized_loadings(self):
    u = np.array([[3, 4], [5, 12]])
    v = np.array([[7, 24], [8, 15]])
    w = np.array([[0, 1, 0], [1, 0, 0]])

    mags, [nu, nv, nw] = normalized_loadings([u, v, w])

    # test magnitudes
    np.testing.assert_almost_equal(mags, [5 * 25 * 1, 13 * 17 * 1][::-1])

    # scale one mode, and test for same tensor
    scaled = [nu, nv, nw * mags.reshape(-1, 1)]
    T1 = para_compose(ribs([u, v, w]))
    T2 = para_compose(ribs(scaled))
    np.testing.assert_almost_equal(T1, T2)

  def test_parafac_roundtrip(self):
    # create 3-factor tensors with know noise
    T0 = np.random.rand(4, 7, 10)
    l0 = parafac(T0, 2)
    T1 = para_compose(ribs(l0)) + .05 * np.random.randn(*T0.shape)

    # see that parafac converges to the same solution
    n1 = normalized_loadings(l0)
    n2 = normalized_loadings(parafac(T1, 2))
    np.testing.assert_almost_equal(n1[0], n2[0], decimal=1)
    for mi in range(3):
      mi1 = n1[1][mi]
      mi2 = n2[1][mi]
      np.testing.assert_almost_equal(mi1, mi2, decimal=1)
