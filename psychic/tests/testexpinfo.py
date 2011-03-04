import unittest
from ..expinfo import *

class TestExpinfo(unittest.TestCase):
  def test_check_markers(self):
    check_markers({1:'left', 2:'right', 3:'right'}) # no error
    self.assertRaises(AssertionError, check_markers, {'left':1})
    self.assertRaises(AssertionError, check_markers, [1, 2])

  def setUp(self):
    self.exp_dic = dict(marker_to_class = {1:'left'},
      trial_offset=[0, 1],
      baseline_offset = [-.5, 0],
      test_folds = [-1, -1, -1, 1, 1, 4, 4],
      ref_chan = ['R1'],
      meg_chan = ['MEG1'],
      eeg_chan = ['Fp1', 'FP2'],
      eog_chan = ['EOG1', 'EOG2'],
      emg_chan = ['Ex1', 'Ex2'],
      notch = [50],
      amplifier = 'fakeamp',
      lab = 'unittest',
      subject = 'S0',
      note = 'running unit test',
      paradigm = 'P300',
      sug_bands = [[1, 12], [.2, 12]],
      sug_time_offsets = [[-.2, 8]],
      sug_chan = ['Fz Cz Pz Oz'.split()])

  def test_complete(self):
    r = check_expinfo(self.exp_dic)

  def test_wrong_key(self):
    self.assertRaises(AssertionError,
      check_expinfo, dict(marker_to_class={1:'left'}, channels=['Fp1', 'Fp2']))

  def test_missing_essentails(self):
    self.assertRaises(Exception, check_expinfo, 
      dict(trial_offset=[0, 1], notch=[]))
    self.assertRaises(Exception, check_expinfo, 
      dict(marker_to_class={1:'left'}, notch=[]))
    self.assertRaises(Exception, check_expinfo, 
      dict(marker_to_class={1:'left'}, trial_offset=[0, 1]))
    check_expinfo(
      dict(marker_to_class={1:'left'}, trial_offset=[0, 1], notch=[]))

  def test_add_expinfo(self):
    d = DataSet(X=np.random.rand(5, 20), Y=np.ones((1, 20)),
      feat_lab=['chann%d' % i for i in range(5)])
    print d, d.feat_lab

    exp_info = check_expinfo(
      dict(marker_to_class={1:'left'}, trial_offset=[0, 1], notch=[]))
    d2 = add_expinfo(d, exp_info)
    assert d2.extra['exp_info'] == exp_info
