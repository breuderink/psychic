# The 10-20 system is used to label exectrode positions for EEG recordings.
# The main reference for the positions are the inion and nasion, and the left
# and right pre-auricular points (LPA, RPA). Deformations due to projecting the
# 3D positions on a 2D plane make the scheme harder to understand than
# necessary.  It works as follows:
#
# Assume that the head is a unit-sphere. We use two angles to describe the
# position:
# 
# - \theta measures the angle from nasion Nz (front-back), range 0--pi
# - \phi measures the lateral (left-right) angle from vertex Cz, range
#   -pi/2--pi/2
#
# For example: 
#   Nz ~ (0, x) polar, or (0, 1, 0) cartesian
#   Cz ~ (pi/2, 0) polar, or (0, 0, 1) cartesian
#   Iz ~ (pi, x) polar, or (0, -1, 0) cartesian
#   LPA ~ (pi/2, -pi/2) polar, or (-1, 0, 0) cartesian
#   RPA ~ (pi/2, -pi/2) polar, or (-1, 0, 0) cartesian
import numpy as np

def per_to_cart(theta, phi):
  theta = theta/100. * np.pi
  phi = phi/100. * np.pi - np.pi/2
  return np.array([
    np.sin(theta) * np.sin(phi), np.cos(theta), np.sin(theta) * np.cos(phi)])

def project_scalp(xyz, v=3.):
  '''
  Project the sensor locations towards a point *below* the cap with distance v
  on the xy plane.
  '''
  # v/(v+z) = x'/x => x' = xv/(v+z)
  x, y, z = xyz
  return (x*v/(v+z), y*v/(v+z))
  

def gen_10_10():
  SUBST = {'C9':'T9', 'C7':'T7', 'C8':'T8', 'C10':'T10',
    'CP9':'TP9', 'CP7':'TP7', 'CP8':'TP8', 'CP10':'TP10',
    'FC9':'FT9', 'FC7':'FT7', 'FC8':'FT8', 'FC10':'FT10'}
  BLACKLIST = ['AF9', 'AF10']
  result = [
    ('Nz', 0, 50), 
    ('Fp1', 10, 10), 
    ('Fpz', 10, 50),
    ('Fp2', 10, 90),
    ('O1', 90, 10),
    ('Oz', 90, 50),
    ('O2', 90, 90),
    ('I1', 90, 0),
    ('Iz', 100, 50),
    ('I2', 90, 100),
    ]
  lr_pos = [str(i) for i in [9, 7, 5, 3, 1, 'z', 2, 4, 6, 8, 10]]
  fb_slab = ['AF', 'F', 'FC', 'C', 'CP', 'P', 'PO']
  for theta_per, slab in zip(np.linspace(20, 80, 7), fb_slab):
    for phi_per, pos in zip(np.linspace(0, 100, 11), lr_pos):
      label = slab + pos
      label = SUBST.get(label, label)
      result.append((label, theta_per, phi_per))
  return [r for r in result if r[0] not in BLACKLIST]

import unittest, matplotlib.pyplot as plt

class Test10_10(unittest.TestCase):
  def test_plot_locs(self):
    locs = []
    for (label, coord) in LAYOUT.items():
      (x, y, z) = coord
      x, y = project_scalp((x, y, z))
      plt.text(x, y + .03, label, fontsize=8, ha='center')
      locs.append((x, y))
    locs = np.asarray(locs)

    plt.plot(locs[:, 0], locs[:, 1], 'ko')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.show()
    
LAYOUT = dict([(l, per_to_cart(theta, phi))
  for (l, theta, phi) in gen_10_10()])

class TestPolar(unittest.TestCase):
  def test_extremes(self):
    np.testing.assert_almost_equal(per_to_cart(0, 50), [0, 1, 0])
    np.testing.assert_almost_equal(per_to_cart(50, 50), [0, 0, 1])
    np.testing.assert_almost_equal(per_to_cart(100, 50), [0, -1, 0])
    np.testing.assert_almost_equal(per_to_cart(50, 0), [-1, 0, 0])
    np.testing.assert_almost_equal(per_to_cart(50, 100), [1, 0, 0])

if __name__ == '__main__':
  unittest.main()
