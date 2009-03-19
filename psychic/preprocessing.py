import logging
import numpy as np
from scipy.signal import cheby1, firwin, lfilter

prep_log = logging.getLogger('psychic.preprocessing')

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

def car(frames, selector=None):
  '''
  Calculate Comman Average Reference. Used to remove distant sources from EEG.
  '''
  if selector==None:
    selector = np.arange(frames.shape[1])
  return frames - np.mean(frames[:, selector], axis=1).reshape(frames.shape[0], 1)

def sliding_window_indices(window_size, window_step, sig_len):
  '''Returns indices for a sliding window with shape [nwindows x window_size]'''
  nwindows = int(np.floor((sig_len - window_size + window_step) / 
    float(window_step)))
  starts = np.arange(nwindows).reshape(nwindows, 1) * window_step
  return starts + np.arange(window_size)

def sliding_window(signals, window_size, window_step):
  '''Replaces the last axis with a window axis and a signal axis'''
  # TODO: use broadcasting magic:
  # np.array([0, 100]).reshape(2, 1, 1) + wi.reshape(1, 3, 5)
  # where wi is created using sliding_window_indices()
  old_shape = signals.shape
  signals = signals.reshape(-1, old_shape[-1]) # to 2D

  indices = sliding_window_indices(window_size, window_step, signals.shape[-1])
  result = []
  for s in signals: 
    curr_wins = s.take(indices=indices)
    result.append(curr_wins)
  result = np.asarray(result)
  return result.reshape(list(old_shape[:-1]) + list(result.shape[-2:]))

def stft(signals, nfft, stepsize):
  '''Calculate the short-time Fourier transform (STFT)'''
  wins = sliding_window(signals, nfft, stepsize) * np.hanning(nfft)
  return np.fft.rfft(wins, axis=signals.ndim)

def spectrogram(signal, nfft, stepsize):
  '''Calculate a spectrogram using the STFT. Returns [frames x frequencies]'''
  # abs is the *magnitude* of a complex number
  return np.abs(stft(signal, nfft, stepsize))

def slice(frames, event_indices, offsets):
  '''
  Slice function, used to extract snippets of EEG from a recording.
  '''
  slices = []
  off_start, off_end = offsets
  for ei in event_indices:
    start, end = ei + off_start, ei + off_end
    if start < 0 or end > frames.shape[0]:
      prep_log.warning('Cannot extract slice [%d, %d]' % (start, end))
    else:
      slices.append(frames[start:end, :])
  return np.concatenate(slices).reshape(len(slices), -1, frames.shape[1])
