import unittest
import numpy as np
from golem import DataSet
from ..nodes import CAR

class TestCar(unittest.TestCase):
  def setUp(self):
    xs = np.arange(40 * 3).reshape(40, 3).astype(float)
    ys = np.ones((40, 1))
    self.d = DataSet(xs=xs, ys=ys, feat_lab=['a', 'b', 'c'])

  def test_car(self):
    d = self.d
    car = CAR()
    cd = car.test(d)

    np.testing.assert_equal(cd.xs.mean(axis=1), np.zeros(d.ninstances))
    np.testing.assert_equal(cd.xs.mean(axis=0), [-1, 0, 1])

  def test_store_car(self):
    d = self.d
    car = CAR(add_CAR=True)
    cd = car.test(d)

    np.testing.assert_equal(cd.xs[:, :-1].mean(axis=1), np.zeros(d.ninstances))
    np.testing.assert_equal(cd.xs[:, -1], np.arange(1, 118+1, 3))
    self.assertEqual(cd.feat_lab[-1], 'CAR')

  def test_mask(self):
    d = self.d
    mask = [1, 2]
    car = CAR(mask=mask)
    cd = car.test(d)
    np.testing.assert_equal(cd.xs[:, mask].mean(axis=1), np.zeros(d.ninstances))
    np.testing.assert_equal(np.mean(cd.xs, 0)[1:], [-.5, .5])
