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

def specgram(signal, NFFT, stepsize):
  '''
  Calculate the FFT of a signal using a sliding Hanning Window.
  '''
  assert(signal.ndim == 1)
  wins = sliding_window(signal, NFFT, stepsize) * np.hanning(NFFT)
  return np.abs(np.fft.rfft(wins, axis=1)).T ** 2
  #@@ np.abs(s.T) ** 2 -> looks like specgram, np.log10 for smooth plot
    

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
