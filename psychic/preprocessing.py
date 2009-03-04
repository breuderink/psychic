import logging
import numpy as np

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

def car(frames):
  '''
  Calculate Comman Average Reference. Used to remove far away sources from EEG.
  '''
  return frames - np.mean(frames, axis=1).reshape(frames.shape[0], 1)

def sliding_window(signal, window_size, window_step):
  '''
  Take a single signal, and move a sliding window over this signal.
  returns a 2D array (windows x signal)
  '''
  assert(signal.ndim == 1)
  nwindows = int(np.floor((len(signal) - window_size + window_step) / \
    float(window_step)))
  starts = np.arange(nwindows).reshape(nwindows, 1) * window_step
  indices = starts + np.arange(window_size)
  return signal.take(indices=indices)

def stft(signal, nfft, stepsize):
  ''' Calculate the short-time Fourier transform (STFT) '''
  assert(signal.ndim == 1)
  wins = sliding_window(signal, nfft, stepsize) * np.hanning(nfft)
  return np.fft.rfft(wins, axis=1)

def spectrogram(signal, nfft, stepsize):
  ''' Calculate a spectrogram using the STFT. Returns (frames x frequencies) '''
  return np.abs(stft(signal, nfft, stepsize))

def slice(frames, event_indices, post_frames, pre_frames=0):
  '''
  Slice function, used to extract snippets of EEG from a recording.
  '''
  slices =[]
  for ei in event_indices:
    start, end = ei-pre_frames, ei+post_frames
    if start < 0 or end > frames.shape[0]:
      prep_log.warning('Cannot extract slice [%d, %d]' % (start, end))
    else:
      slices.append(frames[start:end, :])
  return np.concatenate(slices).reshape(len(slices), -1, frames.shape[1])
