import numpy as np
from scipy import signal
from golem import DataSet
from markers import resample_markers

def ewma_filter(alpha):
  '''
  Filter coefficients for a recursive exponentially weighed moving average
  '''
  alpha = float(alpha)
  assert 0 <= alpha <= 1
  b, a = [1 - alpha], [1, -alpha]
  return b, a

def ewma(x, alpha, v0=0):
  '''
  Causal exponential moving average implemented using scipy.signal.lfilter.
  With alpha as the forgetting factor close to one, x the signal to filter.
  Optionally, an initial estimate can be provided with the float v0.
  '''
  b, a = ewma_filter(alpha)
  x = np.atleast_1d(x).flatten()
  v0 = float(v0)

  zi = signal.lfiltic(b, a, [v0])
  return signal.lfilter(b, a, x, zi=zi)[0]

def ma(x, n):
  '''Causal moving average filter, with signal x, and window-length n.'''
  n = int(n)
  return np.convolve(x, np.ones(n)/n)[:x.size]


def filtfilt_rec(d, (b, a)):
  '''
  Apply a filter defined by the filter coefficients (b, a) to a 
  DataSet, *forwards and backwards*. 
  d.xs is interpreted as [frames x channels].
  '''
  return DataSet(xs=np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), 0,
    d.xs), default=d)

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
  ids = np.ascontiguousarray(d.ids[::factor,:]).astype(d.ids.dtype)

  # construct new DataSet
  return DataSet(xs=xs, ys=ys, ids=ids.reshape(-1, 1), default=d)
