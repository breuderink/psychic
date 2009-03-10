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

def car(frames):
  '''
  Calculate Comman Average Reference. Used to remove far away sources from EEG.
  '''
  return frames - np.mean(frames, axis=1).reshape(frames.shape[0], 1)

def sliding_window(signal, window_size, window_step, axis=0):
  '''
  Take a single signal, and move a sliding window over this signal.
  returns a 2D array (windows x signal)
  '''
  assert signal.ndim == 1, '1D signal required for sliding window.'
  nwindows = int(np.floor((len(signal) - window_size + window_step) /
    float(window_step)))
  starts = np.arange(nwindows).reshape(nwindows, 1) * window_step
  indices = starts + np.arange(window_size)
  return signal.take(indices=indices)

def stft(signal, nfft, stepsize):
  ''' Calculate the short-time Fourier transform (STFT) '''
  assert signal.ndim == 1, '1D signal required for STFT'
  wins = sliding_window(signal, nfft, stepsize) * np.hanning(nfft)
  return np.fft.rfft(wins, axis=1)


def spectrogram(signal, nfft, stepsize):
  ''' Calculate a spectrogram using the STFT. Returns (frames x frequencies) '''
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


def decimate(x, q, n=None, ftype='iir', axis=-1):
  """downsample the signal x by an integer factor q, using an order n filter
  
  By default, an order 8 Chebyshev type I filter is used or a 30 point FIR 
  filter with hamming window if ftype is 'fir'.

  (port to python of the GNU Octave function decimate.)

  Inputs:
      x -- the signal to be downsampled (N-dimensional array)
      q -- the downsampling factor
      n -- order of the filter (1 less than the length of the filter for a
           'fir' filter)
      ftype -- type of the filter; can be 'iir' or 'fir'
      axis -- the axis along which the filter should be applied
  
  Outputs:
      y -- the downsampled signal
  """
  if type(q) != type(1):
    raise Error, "q should be an integer"

  if n is None:
    n = 30 if ftype == 'fir' else 8
  if ftype == 'fir':
    b = firwin(n+1, 1./q, window='hamming')
    y = lfilter(b, 1., x, axis=axis)
  else:
    (b, a) = cheby1(n, 0.05, 0.8/q)
    y = lfilter(b, a, x, axis=axis)

  return y.swapaxes(0,axis)[::q].swapaxes(0,axis)
