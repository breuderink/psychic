import logging, glob
import golem, psychic
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def var_feat(x):
  return np.var(x, axis=0)

def log_feat(x):
  return np.log(x)

def window(x):
  '''Multiply a Hanning window with the time series'''
  return np.hanning(x.shape[0]).reshape(-1, 1) * x

# Setup logging levels
logging.basicConfig(level=logging.WARNING)
logging.getLogger('golem.nodes.ModelSelect').setLevel(logging.INFO)

# Define preprocessing pipeline
preprocessing = golem.nodes.Chain([
  # Downsample with a factor of four
  psychic.nodes.Decimate(4, max_marker_delay=3),
  # Filter to beta range (8--30 Hz)
  psychic.nodes.Filter(lambda s : signal.iirfilter(6, [8./(s/2), 30./(s/2)])),
  # Extract 2 second window centered on key-press
  psychic.nodes.Slice({1:'left', 2:'right', 3:'left', 4:'right'}, [-1, 1]),
  ])

  
# Load file, extract relevant segment, and reduce sample rate
d = psychic.bdf_dataset('S9.bdf')
print d
# Remove all non-EEG channels:
d = golem.DataSet(xs=d.xs[:, :32], feat_lab=d.feat_lab[:32], default=d)
print d, d.feat_lab
print psychic.get_samplerate(d)

preprocessing.train(d)
d = preprocessing.test(d)

print 'After preprocessing:', d

NTRAIN = 500
d, dtest = d[:NTRAIN], d[NTRAIN:]


def I_crit(d, n):
  '''
  Performs 5-fold cross-validation of node n on dataset d, and returns mean
  mutual information. Used for model selection.
  '''
  return golem.loss.mean_std(golem.loss.I,
    golem.cv.cross_validate(golem.cv.seq_splits(d, 5), n))[0]

svm = golem.nodes.ModelSelect(
    [golem.nodes.SVM(C=C) for C in np.logspace(-2, 5, 6)], I_crit)

cl = golem.nodes.Chain([
  golem.nodes.FeatMap(window),
  psychic.nodes.CSP(m=6),
  golem.nodes.FeatMap(var_feat),
  golem.nodes.FeatMap(log_feat),
  svm])

# Build classifier
cl.train(d)

# Do predictions, and split in 5 chronologically disjoint sets
pred = cl.test(dtest)
pred = golem.cv.seq_splits(pred, 5)

def itr(d):
  rate = d.ninstances / (np.max(d.ids) - np.min(d.ids))
  return 60. * golem.loss.I(d) * rate

itrs = np.array([itr(p) for p in pred])
accs = np.array([golem.loss.accuracy(p) for p in pred])
aucs = np.array([golem.loss.auc(p) for p in pred])

print 'ITR: %s -> %.2f' % (itrs.round(2), np.mean(itrs))
print 'Acc: %s -> %.2f' % (accs.round(2), np.mean(accs))
print 'AUC: %s -> %.2f' % (aucs.round(2), np.mean(aucs))
