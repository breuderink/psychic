import numpy as np
from golem import DataSet

class CAR:
  def __init__(self, mask=None, add_CAR=False):
    self.add_CAR = add_CAR
    self.mask = mask

  def train(self, d):
    pass

  def test(self, d):
    mask = self.mask if self.mask != None else np.ones(d.nfeatures, np.bool)
    xs = d.xs.copy() 
    ca = np.mean(xs[:, mask], axis=1).reshape(-1, 1)
    xs[:, mask] -= ca
    feat_lab = d.feat_lab
    if self.add_CAR:
      xs = np.hstack([xs, ca]) # we add the common average again
      feat_lab.append('CAR')
    return DataSet(xs=xs, feat_lab=feat_lab, default=d)
