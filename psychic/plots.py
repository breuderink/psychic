import matplotlib.pyplot as plt
import numpy as np
import scalpplot
from scalpplot import plot_scalp, BIOSEMI_32_LOCS
from scipy import signal

def plot_timeseries(frames, time=None, offset=None, color='k', linestyle='-'):
  frames = np.asarray(frames)
  if offset == None:
    offset = np.max(np.std(frames, axis=0)) * 3
  if time == None:
    time = np.arange(frames.shape[0])
  plt.plot(time, frames - np.mean(frames, axis=0) + 
    np.arange(frames.shape[1]) * offset, color=color, ls=linestyle)

def plot_filt_char(b, a, Fs=1):
  '''Plot frequency, phase and impulse response of a filter (b, a)'''
  (w, h) = signal.freqz(b, a)
  w *= Fs/(2 * np.pi)
  plt.subplot(221)
  plt.title('Frequency response')
  plt.ylabel('Magnitude (dB)')
  plt.plot(w, 2 * 10 * np.log10(np.abs(h)))
  plt.xlim([min(w), max(w)])
  plt.grid()

  plt.subplot(222)
  plt.title('Phase response')
  plt.ylabel('Phase (radians)')
  plt.plot(w, np.unwrap(np.angle(h)))
  plt.xlim([min(w), max(w)])
  plt.grid()

  plt.subplot(223)
  plt.title('Frequency response')
  plt.ylabel('Amplitude')
  plt.plot(w, np.abs(h))
  plt.xlim([min(w), max(w)])
  plt.grid()

  plt.subplot(224)
  plt.title('Impulse reponse')
  IMPULSELEN = 1000
  x = np.arange(IMPULSELEN) - 50
  y = np.zeros(IMPULSELEN)
  y[x == 0] = 1
  plt.plot(x, signal.lfilter(b, a, y))
  plt.ylim([-1, 1])
  plt.xlim([min(x), max(x)])
  plt.grid()

def plot_scalpgrid(scalps, sensors, locs=BIOSEMI_32_LOCS, width=None, 
  clim=None, cmap=None, titles=None):
  '''
  Plots a grid with scalpplots. Scalps contains the different scalps in the
  rows, sensors contains the names for the columns of scalps, locs is a dict
  that maps the sensor-names to locations.

  Width determines the width of the grid that contains the plots. Cmap selects
  a colormap, for example plt.cm.RdBu_r is very useful for AUC-ROC plots.
  Clim is a list containing the minimim and maximum value mapped to a color.

  Titles is an optional list with titles for each subplot.

  Returns a list with subplots for further manipulation.
  '''
  scalps = np.asarray(scalps)
  assert scalps.ndim == 2
  nscalps = scalps.shape[0]
  subplots = []

  if not width:
    width = int(min(8, np.ceil(np.sqrt(nscalps))))
  height = int(np.ceil(nscalps/float(width)))

  if not clim:
    clim = [np.min(scalps), np.max(scalps)]

  plt.clf()
  for i in range(nscalps):
    subplots.append(plt.subplot(height, width, i + 1))
    plot_scalp(scalps[i], sensors, locs, clim=clim, cmap=cmap,
      plot_sensors=False)
    if titles:
      plt.title(titles[i])

  # plot colorbar next to last scalp
  bb = plt.gca().get_position()
  plt.colorbar(cax=plt.axes([bb.xmax + bb.width/10, bb.ymin, bb.width/10,
    bb.height]), ticks=np.linspace(clim[0], clim[1], 5).round(2))

  return subplots
