import unittest
import pylab
import numpy as np
from .. import plots

class TestPlots(unittest.TestCase):
  def test_timeseries(self):
    pylab.figure()
    eeg = np.sin(2 * np.pi * 
      (np.linspace(0, 1, 100).reshape(-1, 1) * np.arange(6)))
    plots.plot_timeseries(eeg, spacing=2)
    pylab.savefig('timeseries.eps')

  def test_scalpplot(self):
    pylab.figure()
    sensors = plots.scalpplot.BIOSEMI_32_LOCS.keys()
    activity = np.random.randn(len(sensors)) * 0.1
    activity[sensors.index('Cz')] = 1
    activity[sensors.index('Fp1')] = 1
    activity[sensors.index('C3')] = -1
    activity[sensors.index('C4')] = -1
    plots.plot_scalp(activity, sensors, plots.scalpplot.BIOSEMI_32_LOCS)
    pylab.savefig('topo.eps')
