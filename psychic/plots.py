import pylab
import numpy as np
import scalpplot
from scalpplot import plot_scalp

def plot_timeseries(frames, spacing=50):
  pylab.plot(frames - np.mean(frames, axis=0) + 
    np.arange(frames.shape[1]) * spacing)
