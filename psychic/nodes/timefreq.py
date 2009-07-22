import logging
import numpy as np
from golem import DataSet
from ..utils import popcorn, spectrogram, sliding_window

class TFC:
  def __init__(self, nfft, win_step):
    self.nfft, self.win_step = nfft, win_step

  def train(self, d):
    pass
  
  def test(self, d):
    assert len(d.feat_shape) == 2 # [frames x channels]
    if d.feat_shape != None:
      assert d.feat_dim_lab[0] == 'time'

    def tf(signal):
      return spectrogram(signal, self.nfft, self.win_step)

    itfc = popcorn(tf, 1, d.nd_xs)
    xs = itfc.reshape(itfc.shape[0], -1)
    feat_shape = itfc.shape[1:]
    feat_dim_lab = ['time', 'frequency', d.feat_dim_lab[1]]

    if d.feat_nd_lab != None:
      old_time = np.asarray([float(i) for i in d.feat_nd_lab[0]])
      time = np.mean(sliding_window(old_time, self.nfft, self.win_step), axis=1)
      time = ['%.1f' % i for i in time]
      dt = np.mean(np.diff(old_time))
      dt = (np.max(old_time) - np.min(old_time)) / old_time.size
      freqs = np.fft.fftfreq(self.nfft, dt) 
      freqs = ['%d' % abs(i) for i in freqs[:self.nfft/2 + 1]]
      channels = d.feat_nd_lab[1]

      feat_nd_lab = [time, freqs, channels]
    else:
      feat_nd_lab = None
    return DataSet(xs=xs, feat_shape=feat_shape, feat_dim_lab=feat_dim_lab, 
      feat_nd_lab=feat_nd_lab, default=d)
