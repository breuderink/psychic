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
  '''
  Calculate the short-time Fourier transform (STFT)
  '''
  assert(signal.ndim == 1)
  wins = sliding_window(signal, nfft, stepsize) * np.hanning(nfft)
  return np.fft.rfft(wins, axis=1)

def spectrogram(signal, nfft, stepsize):
  '''
  Calculate a spectrogram using the STFT. Returns (frames x frequencies)
  '''
  return np.abs(stft(signal, nfft, stepsize))

def slice(frames, event_indices, post_frames, pre_frames=0):
  '''
  Slice function, used to extract snippets of EEG from a recording.
  '''
  slices =[]
  for ei in event_indices:
    slices.append(frames[ei-pre_frames:ei+post_frames, :])
  return np.concatenate(slices).reshape(len(slices), -1, frames.shape[1])

def trialize(time_channels, trial_starts, length):
  trials = []
  for s in trial_starts:
    trial = time_channels[s:s+length, :]
    trials.append(trial)
  return trials

def dwt_features(trial):
  features = []
  for chann_i in range(trial.shape[1]):
    signal = trial[:, chann_i]
    wd = pywt.wavedec(signal, 'db3', level= 7)
    wf = np.vstack([l[0] for l in wd])
    features.append(wf)
  chann_wd = np.vstack(features)
  return chann_wd
