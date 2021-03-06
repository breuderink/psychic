import logging, operator, warnings
import numpy as np
from golem import DataSet, helpers
from scipy import signal

from markers import markers_to_events, biosemi_find_ghost_markers

def sliding_window_indices(window_size, window_step, sig_len):
  '''Returns indices for a sliding window with shape [nwindows x window_size]'''
  nwindows = int(np.floor((sig_len - window_size + window_step) / 
    float(window_step)))
  starts = np.arange(nwindows).reshape(nwindows, 1) * window_step
  return starts + np.arange(window_size)

def sliding_window(signal, window_size, window_step, win_func=None):
  '''Apply a sliding window to a 1D signal. Returns [#windows x window_size].'''
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
  '''
  Calculate a spectrogram using the STFT. 
  Returns [windows x frequencies], in units related to power.
  Equivalent to power spectral density.
  '''
  spec = stft(signal, nfft, stepsize)

  # convert to power. The abs() is the magnitude of a complex number
  spec = np.abs(spec) ** 2 / nfft

  # compensate for missing negative frequencies
  spec[:, 1:-1] *= 2 

  # correct for window
  spec /= np.mean(np.abs(np.hanning(nfft)) ** 2)

  # compensate for overlapping windows
  nwins = spec.shape[0]
  overlap = stepsize / float(nfft)
  spec *= (1 + (nwins - 1) * overlap) / nwins
  return spec

def get_samplerate(d):
  '''Derive the sample rate from the timestamps d.I[0]'''
  return int(np.round(1./np.median(np.diff(d.I[0]))))

def slice(d, markers_to_class, offsets):
  '''
  Slice function, used to extract fixed-length segments of EEG from a recording.
  Returns [segment x frames x channel]
  '''
  assert len(d.feat_shape) == 1
  assert d.nclasses == 1
  start_off, end_off = offsets
  xs, ys, ids = [], [], []

  feat_shape = (end_off - start_off,) + d.feat_shape

  cl_lab = sorted(set(markers_to_class.values()))
  events, events_i = markers_to_events(d.ys.flat)
  for (mark, cl) in markers_to_class.items():
    cl_i = cl_lab.index(cl)
    for i in events_i[events==mark]: # fails if there is *ONE* event
      (start, end) = i + start_off, i + end_off
      if start < 0 or end > d.ninstances:
        logging.getLogger('psychic.utils.slice').warning(
          'Cannot extract slice [%d, %d] for class %s' % (start, end, cl))
        continue
      dslice = d[start:end]
      xs.append(dslice.xs)
      ys.append(cl_i)
      ids.append(d.ids[i, :])

  m = len(xs)
  xs = np.asarray(xs).reshape(m, np.prod(feat_shape))
  ys = np.asarray(ys)
  ys = helpers.to_one_of_n(ys.T, class_rows=range(len(cl_lab))).T
  ids = np.asarray(ids).reshape(m, d.ids.shape[1])

  event_time = np.arange(start_off, end_off) / float(get_samplerate(d))
  time_lab = ['%.3f' % ti for ti in event_time]
  feat_nd_lab = [time_lab, d.feat_lab if d.feat_lab 
    else ['f%d' % i for i in range(d.nfeatures)]]
  feat_dim_lab = ['time', 'channels']
  d = DataSet(xs=xs, ys=ys, ids=ids, cl_lab=cl_lab, 
    feat_shape=feat_shape, feat_nd_lab=feat_nd_lab, 
    feat_dim_lab=feat_dim_lab, default=d)
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

def wolpaw_bitr(N, P):
  assert 0 <= P <= 1
  assert 2 <= N
  result = np.log2(N)
  if P > 0: 
    result += P * np.log2(P)
  if P < 1:
    result += (1 - P) * np.log2((1 - P)/(N - 1.))
  return result
