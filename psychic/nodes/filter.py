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
