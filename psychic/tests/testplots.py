import unittest, os.path
import matplotlib.pyplot as plt
import numpy as np
from .. import plots
from ..positions import POS_10_5

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
    sensors = POS_10_5.keys()
    activity = np.random.randn(len(sensors)) * 1e-3
    activity[sensors.index('Fp1')] = 1.
    activity[sensors.index('C3')] = -1.
    activity[sensors.index('C4')] = 1.
    plots.plot_scalp(activity, sensors, POS_10_5)
      
    plt.savefig(os.path.join('out', 'topo.eps'))

  def test_scalpgrid(self):
    plt.clf()
    sensors = ('Fp1 Fp2 AF3 AF4 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T7 C3 Cz C4 T8' +
      ' CP5 CP1 CP2 CP6 P7 P3 Pz P4 P8 PO3 PO4 O1 Oz O2').split()
    plots.plot_scalpgrid(np.eye(32)[:20], sensors, titles=sensors[:20],
      width=6, cmap=plt.cm.RdBu_r, clim=[-1, 1])
    plt.savefig(os.path.join('out', 'scalpgrid.eps'))
