import pylab
import numpy as np
import scalpplot
from scalpplot import plot_scalp
from scipy import signal

def plot_timeseries(frames, spacing=None, color='k', linestyle='-'):
  if spacing == None:
    spacing = np.max(np.std(frames, axis=0)) * 2
  pylab.plot(frames - np.mean(frames, axis=0) + 
    np.arange(frames.shape[1]) * spacing, color=color, ls=linestyle)

def plot_filt_char(b, a, Fs=1):
  '''Plot frequency, phase and impulse response of a filter (b, a)'''
  (w, h) = signal.freqz(b, a)
  w *= Fs/(2 * np.pi)
  pylab.subplot(221)
  pylab.title('Frequency response')
  pylab.ylabel('Magnitude (dB)')
  pylab.plot(w, 2 * 10 * np.log10(np.abs(h)))
  pylab.xlim([min(w), max(w)])
  pylab.grid()

  pylab.subplot(222)
  pylab.title('Phase response')
  pylab.ylabel('Phase (radians)')
  pylab.plot(w, np.unwrap(np.angle(h)))
  pylab.xlim([min(w), max(w)])
  pylab.grid()

  pylab.subplot(223)
  pylab.title('Frequency response')
  pylab.ylabel('Amplitude')
  pylab.plot(w, np.abs(h))
  pylab.xlim([min(w), max(w)])
  pylab.grid()

  pylab.subplot(224)
  pylab.title('Impulse reponse')
  IMPULSELEN = 1000
  x = np.arange(IMPULSELEN) - 50
  y = np.zeros(IMPULSELEN)
  y[x == 0] = 1
  pylab.plot(x, signal.lfilter(b, a, y))
  pylab.ylim([-1, 1])
  pylab.xlim([min(x), max(x)])
  pylab.grid()
