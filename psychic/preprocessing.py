import logging
import numpy as np

def status_to_events(status_array):
  '''
  Use the lowest 16 bits to extract events from the status channel.
  Events are encoded as TTL pulses, no event is indicated with the value 0.
  Returns (events, indices)
  '''
  status = np.asarray(status_array, int) & 0xffff # oh I love Python...
  change_ids = np.nonzero(np.concatenate([[1], np.diff(status)]))[0]
  events = status[change_ids]
  return (events[np.nonzero(events)], change_ids[np.nonzero(events)])

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


def slice(frames, event_indices, offsets):
  '''
  Slice function, used to extract fixed-length snippets of EEG from a recording.
  Returns [snippet x frames x channel]
  '''
  slices = []
  off_start, off_end = offsets
  assert off_start < off_end
  for ei in event_indices:
    start, end = ei + off_start, ei + off_end
    if start < 0 or end > frames.shape[0]:
      logging.getLogger('psychic.preprocessing').warning(
        'Cannot extract slice [%d, %d]' % (start, end))
    else:
      slices.append(frames[start:end, :])
  return np.concatenate(slices).reshape(len(slices), -1, frames.shape[1])
