import unittest, os
import numpy as np
import matplotlib.pyplot as plt
from ..positions import *

class Test10_5(unittest.TestCase):
  def test_dists(self):
    def dist(a, b):
      return np.linalg.norm((np.atleast_1d(a) - np.atleast_1d(b)))

    def test_eq_dists(labs):
      dists = [dist(POS_10_5[a], POS_10_5[b]) for a, b in zip(labs, labs[1:])]
      self.assert_(np.all(np.abs(np.diff(dists)) < 1e-3))

    test_eq_dists('Nz Fpz AFz Fz FCz Cz CPz Pz POz Oz Iz'.split())
    test_eq_dists('Fpz Fp1 AF7 F7 FT7 T7 TP7 P7 PO7 O1 Oz'.split())
    test_eq_dists('Fpz Fp2 AF8 F8 FT8 T8 TP8 P8 PO8 O2 Oz'.split())
    test_eq_dists('T9 T7 C5 C3 C1 Cz C2 C4 C6 T8 T10'.split())

  def test_plot_locs(self):
    locs = []
    plt.clf()
    for (label, coord) in POS_10_5.items():
      x, y = project_scalp(*coord)
      plt.text(x, y + .03, label, fontsize=6, ha='center')
      locs.append((x, y))
    locs = np.asarray(locs)

    plt.plot(locs[:, 0], locs[:, 1], '.k', ms=3)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.savefig(os.path.join('out', '10-5.eps'))

if __name__ == '__main__':
  unittest.main()
