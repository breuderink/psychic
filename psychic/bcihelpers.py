import numpy as np

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
      raise Exception('Cannot extract slice [%d, %d]' % (start, end))
    else:
      slices.append(frames[start:end, :])
  return np.concatenate(slices).reshape(len(slices), -1, frames.shape[1])
