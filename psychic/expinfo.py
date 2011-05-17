import operator
from golem import DataSet
from golem.nodes import BaseNode
import numpy as np

class ExperimentInfo:
  def __init__(self, ac_freq=None, amplifier=None, lab=None, subject=None,
    note='', eeg_chan=[], eog_chan=[], emg_chan=[], ref_chan=[], 
    experiments={}):
      self.ac_freq=float(ac_freq)
      self.amplifier = str(amplifier)
      self.lab = str(lab)
      self.subject = str(subject)
      self.note = str(note)
      self.eeg_chan = [str(ch) for ch in eeg_chan]
      self.eog_chan = [str(ch) for ch in eog_chan]
      self.emg_chan = [str(ch) for ch in emg_chan]
      self.ref_chan = [str(ch) for ch in ref_chan]

      for (expname, exp) in experiments.items():
        if not set(exp.channels).issubset(self.all_channels):
          raise ValueError('%s is not in record info.' %
            list(set(exp.channels).difference(self.all_channels)))
      self.experiments = experiments

  @property
  def all_channels(self):
    return self.eeg_chan + self.eog_chan + self.emg_chan + self.ref_chan

def markers(markers):
  markers = dict(markers)
  assert len(markers) > 0
  keys = [int(k) for k in markers.keys()]
  values = [str(v) for v in markers.values()]
  assert all([m > 0 for m in keys])
  return dict(zip(keys, values))

def interval(interval):
  assert len(interval) == 2
  assert interval[1] > interval[0]
  return [float(o) for o in interval]

class Experiment:
  def __init__(self, marker_to_class=None, trial_offset=None, 
    baseline_offset=None, test_folds=[], paradigm=None,
    band=None, channels=[]):
    
    self.marker_to_class = markers(marker_to_class)
    self.trial_offset = interval(trial_offset)
    self.baseline_offset = interval(baseline_offset) # baseline is required?
    self.test_folds = [int(tf) for tf in test_folds]
    self.paradigm = str(paradigm)
    self.band = interval(band)
    self.channels = [str(ch) for ch in channels]

def add_expinfo(exp_info, d):
  '''
  Add experiment info to a DataSet d, and perform some sanity checks, i.e.
  checking of matching channel names.
  '''
  if not set(exp_info.all_channels).issubset(d.feat_lab):
    raise ValueError('%s is not in record info.' %
      list(set(exp_info.all_channels).difference(d.feat_lab)))

  extra = {'exp_info' : exp_info}
  extra.update(d.extra)
  return DataSet(extra=extra, default=d)

class OnlyEEG(BaseNode):
  def train_(self, d):
    assert len(d.feat_shape) == 1, 'Please use before extracting trials'
    eeg_chan = d.extra['exp_info']['eeg_chan']
    self.keep_ii = [i for (i, ch) in enumerate(eeg_chan) if ch in eeg_chan]

  def apply_(self, d):
    return DataSet(X=X[self.keep_ii], 
      fealab=[d.fealab[i] for i in self.keep_ii], default=d)
