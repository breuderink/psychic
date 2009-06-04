import numpy as np
from scipy import signal

def fir_bandpass(start, end, transition, Fs=1.):
  '''
  Calculate a FIR bandpass filter using the Remez exchange algorithm.
  Equation 16.3 in [1] is used to caluculate the filter length.
  returns (b, a)
  
  [1] S. W. Smith and Others, "The scientist and engineer's guide 
  to digital signal processing", 1997
  '''
  M = int(4. / (transition / float(Fs)))
  bands = np.asarray([0, start-transition, 
    start, end, end+transition, Fs/2], float)
  gain = [0, 1, 0]
  return (signal.remez(M, bands, gain, type='bandpass', Hz=Fs), 1)

def fbfilter(b, a, xs):
  xs = signal.lfilter(b, a, xs, axis=0) # forward
  return np.flipud(signal.lfilter(b, a, np.flipud(xs), axis=0)) # backward
