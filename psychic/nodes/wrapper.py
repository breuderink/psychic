import numpy as np
from ..utils import slice, get_samplerate
from ..filtering import decimate_rec
from golem.nodes import BaseNode

class Slice(BaseNode):
  def __init__(self, mark_to_cl, offsets):
    '''
    In contrast to psychic.utils.slice, offsets are specified in *seconds*
    '''
    self.mdict, self.offsets = mark_to_cl, np.asarray(offsets)
    BaseNode.__init__(self)

  def train_(self, d):
    self.sample_rate = get_samplerate(d)

  def test_(self, d):
    return slice(d, self.mdict, (self.offsets * self.sample_rate).astype(int))

class Decimate(BaseNode):
  def __init__(self, factor):
    self.factor = factor
    BaseNode.__init__(self)

  def test_(self, d):
    return decimate_rec(d, self.factor)

