import operator
from golem import DataSet
from golem.nodes import BaseNode
import numpy as np

TRIALINFO = ['marker_to_class', 'trial_offset', 'baseline_offset', 
  'test_folds']
CHANNINFO = ['ref_chan', 'meg_chan', 'eeg_chan', 'eog_chan', 'emg_chan']
RECINFO = ['notch', 'amplifier', 'lab', 'subject', 'note']
TASKINFO = ['paradigm', 'sug_bands', 'sug_time_offsets', 'sug_chan']

def check_markers(markers):
  assert isinstance(markers, dict)
  assert all([isinstance(m, int) for m in markers.keys()])
  assert all([isinstance(cl, str) for cl in markers.values()])
  assert len(markers) > 0

def check_expinfo(expinfo):
  valid_keys = TRIALINFO + CHANNINFO + RECINFO + TASKINFO
  for key in expinfo.keys():
    assert key in valid_keys, ('%s is not a valid keyword' % key)
  
  # check markers
  check_markers(expinfo['marker_to_class'])

  # check trial offset
  to = expinfo['trial_offset']
  assert len(to) == 2
  assert np.diff(to) > 0

  # check baseline offset
  bo = expinfo.get('baseline_offset')
  if bo != None:
    assert len(to) == 2
    assert np.diff(to) > 0
  
  # test_folds
  tf = expinfo.get('test_folds')
  if tf != None:
    assert all([isinstance(fold, int) for fold in tf])

  # verify that channels are strings
  for changroup in CHANNINFO + ['sug_chann']:
    chans = expinfo.get(changroup)
    if chans:
      assert all([isinstance(ch, str) for ch in chans])
 
  # verify format of notch
  assert 'notch' in expinfo
  notch = expinfo['notch']
  if notch:
    assert all([isinstance(freq, int) for freq in notch])

  # verify string fields
  for field in ['amp', 'lab', 'subject', 'note', 'paradigm']:
    assert isinstance(expinfo.get(field, ''), str)

  # verify sug_bands and sug_time_intervals
  for sug in ['sug_bands', 'sug_time_offsets']:
    sug_int = expinfo.get(sug)
    if sug_int != None:
      sug_int = np.asarray(sug_int)
      assert sug_int.ndim == 2
      assert np.all(np.diff(sug_int, axis=1) > 0)

  return expinfo

def add_expinfo(exp_dict, d):
  # Testing the test_folds is difficult; markers might be present in short runs
  # and slicing might ignore trials on the boundary.

  # Verify that channels are a subset of feat_lab.
  channels = [exp_dict.get(chgroup, []) for chgroup in CHANNINFO]
  channels.extend(exp_dict.get('sug_chan', []))
  channels = reduce(operator.add, channels)
  for ch in channels:
    assert ch in d.feat_lab, ('%s not found in feat_lab' % ch)

  extra = {'exp_info' : exp_dict}
  extra.update(d.extra)
  return DataSet(extra=extra, default=d)

class AutoReref(BaseNode): 
  pass

class AutoNotch(BaseNode): 
  pass

class SugBandPass(BaseNode): 
  pass

class RemoveEOG(BaseNode): 
  pass

class OnlyEEG(BaseNode):
  def train_(self, d):
    assert len(d.feat_shape), 'Please use before extracting trials'
    eeg_chan = d.extra['exp_info']['eeg_chan']
    self.keep_ii = [i for (i, ch) in enumerate(eeg_chan) if ch in eeg_chan]

  def apply_(self, d):
    return DataSet(X=X[self.keep_ii], 
      fealab=[d.fealab[i] for i in self.keep_ii], default=d)

class AutoExTrials(BaseNode): 
  pass

class AutoBaseline(BaseNode): 
  pass

class EMGEnvelope(BaseNode): 
  pass

class SugChannels(BaseNode): 
  pass

class SugIntervals(BaseNode): 
  pass

def auto_folds(d):
  pass



