import numpy as np
from scipy import signal
from golem import DataSet

class OnlineFilter:
  def __init__(self, b, a):
    self.b, self.a = b, a
    self.zi = []

  def train(self, d):
    pass

  def test(self, d):
    b, a = self.b, self.a
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
