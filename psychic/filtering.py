import numpy as np
from scipy import signal
from golem import DataSet
from markers import resample_markers

def fir_bandpass(start, end, transition, Fs=1.):
  '''
  Calculate a FIR bandpass filter using the Remez exchange algorithm.
  Equation 16.3 in [1] is used to caluculate the filter length.
  returns (b, a)
  
  [1] S. W. Smith and Others, "The scientist and engineer's guide 
  to digital signal processing", 1997
  '''
  M = int(4. / (transition / float(Fs)))
  bands = np.asarray([0, start-transition, 
    start, end, end+transition, Fs/2], float)
  gain = [0, 1, 0]
  return (signal.remez(M, bands, gain, type='bandpass', Hz=Fs), [1])

def filtfilt_rec((b, a), d):
  '''
  Apply a filter defined by the filter coefficients (b, a) to a 
  DataSet, *forwards and backwards*. 
  d.xs is interpreted as [frames x channels].
  '''
  xs = np.hstack([signal.filtfilt(b, a, d.xs[:, i]).reshape(-1, 1) 
    for i in range(d.nfeatures)])
  return DataSet(xs=xs, default=d)

def resample_rec(d, factor, max_marker_delay=0):
  '''Resample a recording to length d.ninstances * factor'''
  new_len = int(d.ninstances * factor)
  ys = resample_markers(d.ys.flatten(), new_len, 
    max_delay=max_marker_delay).reshape(-1, 1)

  # calculate xs and ids
  xs, ids = signal.resample(d.xs, new_len, t=d.ids)
  xs = xs.astype(d.xs.dtype) # keep old dtype

  # construct new DataSet
  extra = d.extra.copy()
  return DataSet(xs=xs, ys=ys, ids=ids.reshape(-1, 1), extra=extra, default=d)

def decimate_rec(d, factor, max_marker_delay=0):
  '''Decimate a recording using an anti-aliasing filter.'''
  assert isinstance(factor, int), 'Decimation factor should be an int'

  # anti-aliasing filter
  (b, a) = signal.iirfilter(8, .8 / factor, btype='lowpass', rp=0.05, 
    ftype='cheby1')
  xs = d.xs.copy()
  for i in range(d.nfeatures):
    xs[:,i] = signal.filtfilt(b, a, xs[:, i])

  xs = np.ascontiguousarray(xs[::factor,:]).astype(d.xs.dtype)

  ys = resample_markers(d.ys.flatten(), xs.shape[0],
    max_delay=max_marker_delay).reshape(-1, 1)

  # calc ids
  ids = np.ascontiguousarray(d.ids[::factor,:]).astype(d.ids.dtype)

  # construct new DataSet
  extra = d.extra.copy()
  return DataSet(xs=xs, ys=ys, ids=ids.reshape(-1, 1), extra=extra, default=d)
