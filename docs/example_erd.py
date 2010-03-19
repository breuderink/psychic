import logging, glob
import golem, psychic
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def var_feat(x):
  return np.var(x, axis=0)

def log_feat(x):
  return np.log(x)

def erd_slope(x):
  return np.apply_along_axis(lambda x: np.abs(signal.hilbert(x)), 0, x)

def window(x):
  '''Multiply a Hanning window with the time series'''
  return np.hanning(x.shape[0]).reshape(-1, 1) * x

# Setup logging levels
logging.basicConfig(level=logging.WARNING)
logging.getLogger('golem.nodes.ModelSelect').setLevel(logging.INFO)

# Define preprocessing pipeline
preprocessing = golem.nodes.Chain([
  psychic.nodes.Decimate(4, max_marker_delay=3),
  psychic.nodes.Filter(lambda s : signal.iirfilter(6, [8./(s/2), 30./(s/2)])),
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

feat_viz = golem.nodes.Chain([
  golem.nodes.FeatMap(window), 
  psychic.nodes.CSP(m=10), 
  golem.nodes.FeatMap(erd_slope)])
feat_viz.train(d)


dhil = feat_viz.test(d)
time = np.asarray(dhil.feat_nd_lab[0], float)
plt.clf()
psychic.plots.plot_timeseries(np.mean(dhil.get_class(0).nd_xs, axis=0), 
  offset=1, color='k', time=time)
psychic.plots.plot_timeseries(np.mean(dhil.get_class(1).nd_xs, axis=0),
  offset=1, color='k', time=time, linestyle='--')
plt.axis('tight')
plt.savefig('hilbert.pdf')


def I_crit(d, n):
  '''
  Performs 5-fold cross-validation of node n on dataset d, and returns mean
  mutual information. Used for model selection.
  '''
  return golem.loss.mean_std(golem.loss.I,
    golem.cv.cross_validate(golem.cv.seq_splits(d, 5), n))[0]

svm = golem.nodes.ModelSelect(
    [golem.nodes.SVM(C=C) for C in np.logspace(-2, 5, 6)], I_crit)

csp_pipe = golem.nodes.Chain([
  golem.nodes.FeatMap(window),
  psychic.nodes.CSP(m=6),
  golem.nodes.FeatMap(var_feat),
  golem.nodes.FeatMap(log_feat),
  svm])

hilbert_pipe = golem.nodes.Chain([
  golem.nodes.FeatMap(window),
  psychic.nodes.CSP(m=8),
  golem.nodes.FeatMap(erd_slope),
  svm])

# Train and evaluate BCI classifier
for cl in [csp_pipe, hilbert_pipe]:
  cl.train(d)
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
