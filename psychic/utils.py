import logging, operator
import numpy as np
from bdfreader import BDFReader
from golem import DataSet, helpers
from scipy import signal

from markers import markers_to_events, biosemi_find_ghost_markers,\
  resample_markers

def sliding_window_indices(window_size, window_step, sig_len):
  '''Returns indices for a sliding window with shape [nwindows x window_size]'''
  nwindows = int(np.floor((sig_len - window_size + window_step) / 
    float(window_step)))
  starts = np.arange(nwindows).reshape(nwindows, 1) * window_step
  return starts + np.arange(window_size)

def sliding_window(signal, window_size, window_step, win_func=None):
  '''Apply a sliding window to a 1D signal'''
  if signal.ndim != 1:
    raise ValueError, 'Sliding window works on 1D arrays only!'
  if win_func != None:
    if win_func.size != window_size:
      raise ValueError, 'window_size (%d) does not match win_func.size (%d)' % (
        window_size, win_func.size)
  indices = sliding_window_indices(window_size, window_step, signal.shape[0])
  windows = signal.take(indices=indices)
  if win_func != None:
    windows = windows * win_func # broadcasting matches from last dim
  return windows

def stft(signal, nfft, stepsize):
  '''Calculate the short-time Fourier transform (STFT).
  Returns [windows x FFT coefficients]'''
  wins = sliding_window(signal, nfft, stepsize, win_func=np.hanning(nfft))
  return np.fft.rfft(wins, axis=1)

def spectrogram(signal, nfft, stepsize):
  '''Calculate a spectrogram using the STFT. Returns [frames x frequencies]'''
  # abs is the *magnitude* of a complex number
  return np.abs(stft(signal, nfft, stepsize))

def popcorn(f, axis, array, *args):
  # array.shape ~ (i, j, k, l), axis = 1
  array = array.swapaxes(axis, -1)
  x_shape = array.shape[:-1]
  # x_shape~ (i, l, k)

  array = array.reshape(-1, array.shape[-1]) 
  # array.shape ~ (x, j)

  result = np.asarray([f(a, *args) for a in array])
  y_shape = result.shape[1:]
  # y_shape ~ (y1, y2, y3)

  result = result.reshape(x_shape + (-1,)) 
  # result.shape ~ (i, l, k, y)
  result = result.swapaxes(axis, -1)
  # result.shape ~ (i, y, k, l)

  final_shape = result.shape[:axis] + y_shape + result.shape[axis+1:]
  result = result.reshape(final_shape)
  # result.shape = (i, y1, y2, y3, k, l)
  return result

def bdf_dataset(fname):
  STATUS = 'Status'
  f = open(fname, 'rb')
  try:
    b = BDFReader(f)
    frames = b.read_all()
  finally:
    f.close()

  data_mask = [i for i, lab in enumerate(b.labels) if lab != STATUS]
  status_mask = b.labels.index(STATUS)
  feat_lab = [b.labels[i] for i in data_mask]
  assert min(b.sample_rate) == max(b.sample_rate)
  sample_rate = b.sample_rate[0]
  ids = (np.arange(frames.shape[0]) / float(sample_rate)).reshape(-1, 1)
  d = DataSet(
    xs=frames[:,data_mask], 
    ys=frames[:,status_mask].reshape(-1, 1).astype(int) & 0xffff, 
    ids=ids, feat_lab=feat_lab, cl_lab=['status'])
  ghosts = biosemi_find_ghost_markers(d.ys.flatten())
  if len(ghosts) > 0:
    logging.getLogger('Psychic.bdf_dataset').warning(
      'Found ghost markers: %s' % str(zip(d.ys.flatten()[ghosts], ghosts)))
  return d

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
  ys = resample_markers(d.ys.flatten(), d.ninstances/factor,
    max_delay=max_marker_delay).reshape(-1, 1)

  # anti-aliasing filter
  (b, a) = signal.iirfilter(8, .8 / factor, btype='lowpass', rp=0.05, 
    ftype='cheby1')
  xs = d.xs.copy()
  for i in range(d.nfeatures):
    xs[:,i] = signal.filtfilt(b, a, xs[:, i])

  xs = np.ascontiguousarray(xs[::factor,:]).astype(d.xs.dtype)

  # calc ids
  ids = np.ascontiguousarray(d.ids[::factor,:]).astype(d.ids.dtype)

  # construct new DataSet
  extra = d.extra.copy()
  return DataSet(xs=xs, ys=ys, ids=ids.reshape(-1, 1), extra=extra, default=d)

def slice(d, marker_dict, offsets):
  '''
  Slice function, used to extract fixed-length snippets of EEG from a recording.
  Returns [snippet x frames x channel]
  '''
  assert len(d.feat_shape) == 1
  start_off, end_off = offsets
  xs, ys, ids = [], [], []
  (events, events_i) = markers_to_events(d.ys.flat)
  for (k, v) in marker_dict.items():
    for i in events_i[events==v]:
      (start, end) = i + start_off, i + end_off
      if start < 0 or end > d.ninstances:
        logging.getLogger('psychic.utils.slice').warning(
          'Cannot extract slice %d-%d for class %s' % (start, end, k))
        continue
      dslice = d[start:end]
      xs.append(dslice.xs)
      ys.append(v)
      ids.append(dslice.ids[0, :])

  xs = np.asarray(xs)
  feat_shape = xs.shape[1:]
  xs = xs.reshape(xs.shape[0], -1)
  event_time = dslice.ids[:, 0] - d[i].ids[0, 0]
  time_lab = ['%.2f' % ti for ti in event_time]
  feat_nd_lab = [time_lab, d.feat_lab if d.feat_lab 
    else ['f%d' % i for i in range(d.nfeatures)]]
  feat_dim_lab = ['time', 'channels']
  ys = helpers.to_one_of_n(ys)
  cl_lab = [lab for lab, _ in sorted(marker_dict.items(), 
    key=operator.itemgetter(1))]
  ids = np.asarray(ids)
  d = DataSet(xs=xs, ys=ys, ids=ids, cl_lab=cl_lab, 
    feat_shape=feat_shape, feat_nd_lab=feat_nd_lab, 
    feat_dim_lab=feat_dim_lab, extra=dict(event_time=event_time))
  return d.sorted()

def find_segments(events, event_indices, start_mark, end_mark):
  '''Helper to find matching start/end markers in an event array'''
  events, event_indices = np.asarray(events), np.asarray(event_indices)
  assert events.size == event_indices.size
  mask = (events==start_mark) | (events==end_mark)
  sevents, sevent_ids = events[mask], event_indices[mask]
  stack, result = [], []
  for si in range(sevent_ids.size):
    if sevents[si] == start_mark:
      stack.append(sevent_ids[si])
    else:
      assert stack != [], 'Missing start marker'
      result.append((stack.pop(), sevent_ids[si]))
  if not stack == []:
    logging.getLogger('psychic.utils.find_segments').warning(
      'Did not end start marker(s) at %s' % repr(stack))
  return result

def cut_segments(d, marker_tuples, offsets=[0, 0]):
  '''
  Cut a dataset into segment using (start_marker, end_marker) tuples.
  Returns a list with DataSets.
  '''
  start_off, end_off = offsets
  segments = []
  e, ei = markers_to_events(d.ys.flat)
  for (sm, em) in marker_tuples:
    segments.extend(find_segments(e, ei, sm, em))
  segments.sort()
  return [d[s + start_off:e + end_off] for (s, e) in segments]
