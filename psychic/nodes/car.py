import numpy as np
from golem import DataSet
from golem.nodes import BaseNode

class CAR(BaseNode):
  def __init__(self, mask=None, add_CAR=False):
    '''
    Mask indicates channels that are used and affected by the CAR.
    Mask can be a boolean array, indices or a slice.
    If add_CAR, the CAR is added as a new channel.
    '''
    BaseNode.__init__(self)
    self.add_CAR = add_CAR
    self.mask = mask

  def apply_(self, d):
    mask = self.mask if self.mask != None else np.ones(d.nfeatures, bool)
    xs = d.xs.copy() 
    ca = np.mean(xs[:, mask], axis=1).reshape(-1, 1)
    xs[:, mask] -= ca
    feat_lab = d.feat_lab
    if self.add_CAR:
      xs = np.hstack([xs, ca]) # we add the CAR as a channel
      feat_lab.append('CAR')
    return DataSet(xs=xs, feat_lab=feat_lab, default=d)
