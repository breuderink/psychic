import numpy as np
from golem.nodes import BaseNode
from golem import DataSet
from spatialfilter import sym_whitening, cov0
from ..utils import get_samplerate
from scipy import signal

class SlowSphering(BaseNode):
  def __init__(self, isi=10, reest=.5):
    '''
    Define a SlowSphering node, with inter-stimulus interval isi in seconds
    which is reestimated every reest seconds.
    '''
    self.isi = isi
    self.reest = reest
    BaseNode.__init__(self)

  def train_(self, d):
    self.samplerate = get_samplerate(d)
    nyq = ((1./self.reest) / 2.)
    self.cutoff = (1./self.isi) / nyq
    self.log.debug('set cutoff: %.3f' % self.cutoff)
    self.fil = signal.iirfilter(4, self.cutoff, btype='low')

  def apply_(self, d):
    xs = slow_sphere(d.xs, self.fil, int(self.reest * self.samplerate))
    return DataSet(xs=xs, default=d)

def slow_sphere(samples, (b, a), wstep):
  '''
  Applies a symmetrical whitening transform to samples, based on locally
  estimated covariance matrices. (b, a) is a FIR or IIR filter that determines
  the type of smoothing, step_size determines how much the window shifts before
  re-estimation of the whitening transform.

  The actual calculation is performed as follows:
  1) a local covariance is estimated for segments of step_size length
  2) the local covariances are forward filtered with (b, a)
  3) each segment is individually whitened with a symmetrical whitening
     transfrom

  The filter (b, a) should be designed to match wstep.
  '''
  samples = np.atleast_2d(samples)

  sigs = np.asarray([cov0(samples[i:i+wstep]) for i in 
    range(0, samples.shape[0], wstep)])

  sigs = signal.lfilter(b, a, sigs, axis=0)

  ws = [sym_whitening(s) for s in sigs]
  return np.vstack([np.dot(samples[i * wstep:(i+1) * wstep], W) 
    for i, W in enumerate(ws)])

