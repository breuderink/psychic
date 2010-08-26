# TODO:
# [ ] add description (old one is invalidated)
# [ ] add extra sensors to 10-10 standard (F9, Ns etc)
# [ ] integrate with scalp-plot functions
# [ ] add (small) sensor dots by default in scalp plot, names optional.
import numpy as np

def per_to_cart(theta, phi):
  theta = theta/100. * np.pi
  phi = phi/100. * np.pi - np.pi/2
  return [np.sin(theta) * np.sin(phi), np.cos(theta), 
    np.sin(theta) * np.cos(phi)]

def project_scalp(xyz, v=2.):
  '''
  Project the sensor locations towards a point *below* the cap with distance v
  on the xy plane.
  '''
  # v/(v+z) = x'/x => x' = xv/(v+z)
  x, y, z = xyz
  return (x*v/(v+z), y*v/(v+z))

def gen_10_10():
  # add irregularly placed sensors
  result = {
    'Fp1': (10, 0), 'Fpz': (0, 50), 'Fp2': (10, 100), 'Nz': (-10, 50),
    'O1' : (110, 0), 'Oz' : (100, 50), 'O2' : (110, 100),
    'I1' : (90, -10), 'Iz' : (100, 50), 'I2' : (90, 110)}

  # generate polar sensor positions with names
  lr_pos = [str(i) for i in [9, 7, 5, 3, 1, 'z', 2, 4, 6, 8, 10]]
  fb_slab = ['AF', 'F', 'FC', 'C', 'CP', 'P', 'PO']
  for theta_per, slab in zip(np.linspace(20, 80, 7), fb_slab):
    for phi_per, pos in zip(np.linspace(-10, 110, 11), lr_pos):
      result[slab + pos] = (theta_per, phi_per)
 
  # convert to cartesian coords
  result = dict([(k, per_to_cart(*v)) for (k, v) in result.items()])
 
  # substitute with sepcialized labels and remove illegal sensors
  SUBST = {'C9':'T9', 'C7':'T7', 'C8':'T8', 'C10':'T10',
    'CP9':'TP9', 'CP7':'TP7', 'CP8':'TP8', 'CP10':'TP10',
    'FC9':'FT9', 'FC7':'FT7', 'FC8':'FT8', 'FC10':'FT10'}
  BLACKLIST = ['AF9', 'AF10']
  result = dict([(SUBST.get(k, k), v) for k, v in result.items()
    if k not in BLACKLIST])

  return result

import unittest, matplotlib.pyplot as plt

class Test10_10(unittest.TestCase):
  def test_plot_locs(self):
    locs = []
    print LAYOUT['Nz']
    print LAYOUT['I1']
    print LAYOUT['Iz']
    print LAYOUT['I2']
    for (label, coord) in LAYOUT.items():
      x, y = project_scalp(coord)
      plt.text(x, y + .03, label, fontsize=8, ha='center')
      locs.append((x, y))
    locs = np.asarray(locs)

    plt.plot(locs[:, 0], locs[:, 1], 'ko')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.show()
    
LAYOUT = gen_10_10()

class TestPolar(unittest.TestCase):
  def test_extremes(self):
    np.testing.assert_almost_equal(per_to_cart(0, 50), np.array([0, 1, 0]))
    np.testing.assert_almost_equal(per_to_cart(50, 50), np.array([0, 0, 1]))
    np.testing.assert_almost_equal(per_to_cart(100, 50), np.array([0, -1, 0]))
    np.testing.assert_almost_equal(per_to_cart(50, 0), np.array([-1, 0, 0]))
    np.testing.assert_almost_equal(per_to_cart(50, 100), np.array([1, 0, 0]))

if __name__ == '__main__':
  unittest.main()
