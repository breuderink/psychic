import pylab
import numpy as np
import topoplot
from topoplot import plot_topo

def plot_timeseries(frames, spacing=50):
  pylab.plot(frames - np.mean(frames, axis=0) + 
    np.arange(frames.shape[1]) * spacing)
