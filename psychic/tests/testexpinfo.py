import unittest
from ..expinfo import *

class TestHelpers(unittest.TestCase):
  def test_interval(self):
    self.assertEqual(interval([1, 2]), [1., 2.])
    self.assertRaises(AssertionError, interval, [0, 0])
    self.assertRaises(AssertionError, interval, [0, -1])

  def test_check_markers(self):
    self.assertEqual(
      markers({1:'left', 2:'right', 3:'right'}),
      {1:'left', 2:'right', 3:'right'})
    self.assertRaises(ValueError, markers, {'marker':1})
    self.assertRaises(TypeError, markers, [1, 2])

class TestExpInfo(unittest.TestCase):
  def setUp(self):
    self.ex_mi = Experiment(marker_to_class={1:'left', 2:'right'},
      trial_offset=[.5, 3], baseline_offset=[-1, 0], band=[8, 30],
      channels=['C3', 'C4'], paradigm='MI', test_folds=range(10))
    
    self.expinfo = ExperimentInfo(
      ac_freq=50, 
      amplifier='BioSemi ActiveTwo', 
      lab='UT/HMI', 
      subject='Fake',
      note='recording note', 
      eeg_chan='C3 Cz C4'.split(),
      eog_chan='L R T B'.split(),
      emg_chan='Ex1 Ex2'.split(),
      ref_chan='Ma1 Ma2'.split(),
      experiments={'LR' : self.ex_mi})


  def test_experiment_info(self):
    self.assertEqual(set(self.expinfo.all_channels),
      set('C3 Cz C4 L R T B Ex1 Ex2 Ma1 Ma2'.split()))

  def test_add_expinfo(self):
    d_bad = DataSet(X=np.random.rand(5, 20), Y=np.ones((1, 20)),
      feat_lab=['chann%d' % i for i in range(5)])
    
    d_good = DataSet(X=np.random.rand(11, 20), Y=np.ones((1, 20)),
      feat_lab=self.expinfo.all_channels)

    d = add_expinfo(self.expinfo, d_good)
    self.assertRaises(ValueError, add_expinfo, self.expinfo, d_bad)
