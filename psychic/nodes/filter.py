import numpy as np
from scipy import signal
from golem import DataSet
from golem.nodes import BaseNode
from psychic.utils import get_samplerate

class Filter(BaseNode):
  def __init__(self, filt_design_func):
    '''
    Forward-backward filtering node. filt_design_func is a function that takes
    the sample rate as an argument, and returns the filter coefficients (b, a).
    '''
    BaseNode.__init__(self)
    self.filt_design_func = filt_design_func

  def train_(self, d):
    fs = get_samplerate(d)
    self.log.info('Detected sample rate of %d Hz' % fs)
    self.filter = self.filt_design_func(fs)

  def apply_(self, d):
    b, a = self.filter
    xs = np.hstack([signal.filtfilt(b, a, d.xs[:, i]).reshape(-1, 1) 
      for i in range(d.nfeatures)])
    return DataSet(xs=xs, default=d)

class OnlineFilter(Filter):
  def __init__(self, filt_design_func):
    Filter.__init__(self, filt_design_func)
    self.zi = []

  def apply_(self, d):
    b, a = self.filter
    if self.zi == []:
      self.zi = [signal.lfiltic(b, a, np.zeros(b.size)) for fi in 
        range(d.nfeatures)]

    new_zi = []
    xs = []
    for i in range(d.nfeatures):
      xi, zii = signal.lfilter(b, a, d.xs[:, i], zi=self.zi[i])
      xs.append(xi.reshape(-1, 1))
      new_zi.append(zii)
    self.zi = new_zi

    return DataSet(xs=np.hstack(xs), default=d)

class Winsorize(BaseNode):
  def __init__(self, cutoff=[.05, .95]):
    self.cutoff = np.atleast_1d(cutoff)
    assert self.cutoff.size == 2
    BaseNode.__init__(self)

  def train_(self, d):
    assert len(d.feat_shape) == 1
    self.lims = np.apply_along_axis(lambda x: np.interp(self.cutoff, 
      np.linspace(0, 1, d.ninstances), np.sort(x)), 0, d.xs)
    
  def apply_(self, d):
    return DataSet(xs=np.clip(d.xs, self.lims[0,:], self.lims[1:]),
      default=d)
