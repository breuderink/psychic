import unittest, os.path
import matplotlib.pyplot as plt
import numpy as np
from .. import plots
from ..scalpplot import BIOSEMI_32_LOCS

class TestPlots(unittest.TestCase):
  def test_timeseries(self):
    plt.clf()
    plots.plot_timeseries(np.random.randn(1000, 10)) 
    plt.savefig(os.path.join('out', 'timeseries.eps'))

  def test_timeseries2(self):
    plt.clf()
    eeg = np.sin(2 * np.pi * 
      (np.linspace(0, 1, 100).reshape(-1, 1) * np.arange(6)))
    plots.plot_timeseries(eeg, time=np.linspace(0, 1, 100), offset=2, 
      color='b', linestyle='--') 
    plt.savefig(os.path.join('out', 'timeseries2.eps'))

  def test_scalpplot(self):
    plt.clf()
    sensors = plots.scalpplot.BIOSEMI_32_LOCS.keys()
    activity = np.random.randn(len(sensors)) * 0.1
    activity[sensors.index('Fp1')] = 1
    activity[sensors.index('C3')] = -1
    activity[sensors.index('C4')] = 1
    plots.plot_scalp(activity, sensors, plots.scalpplot.BIOSEMI_32_LOCS)
      
    plt.savefig(os.path.join('out', 'topo.eps'))

  def test_scalpgrid(self):
    plt.clf()
    sensors = BIOSEMI_32_LOCS.keys()
    plots.plot_scalpgrid(np.eye(32)[:20], sensors, titles=sensors[:20],
      width=6, cmap=plt.cm.RdBu_r, clim=[-1, 1])
    plt.savefig(os.path.join('out', 'scalpgrid.eps'))
