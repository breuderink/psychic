import logging
import golem, psychic
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def var_feat(x):
  '''Calculate band-power by taking the variance of the time dimension'''
  return np.var(x, axis=0)

def window(x):
  '''Multiply a Hanning window with the time series'''
  return np.hanning(x.shape[0]).reshape(-1, 1) * x

# Setup logging levels
logging.basicConfig(level=logging.WARNING)

# Load file, extract relevant segment, and reduce sample rate
d = psychic.bdf_dataset('CL.bdf')
ds = psychic.cut_segments(d, [(219, 220)], [-100, 100]) 
# only the first two block contain imaginary movement
d = ds[0] + ds[1] 

# Define preprocessing pipeline, and process EEG
preprocessing = golem.nodes.Chain([
  psychic.nodes.Decimate(4),
  psychic.nodes.Filter(lambda s : signal.iirfilter(6, [8./(s/2), 30./(s/2)])),
  psychic.nodes.CAR(),
  psychic.nodes.Slice({2:'left', 3:'right'}, [0, 3]),
  golem.nodes.FeatMap(window)
  ])

preprocessing.train(d)
d = preprocessing.test(d)

print 'After preprocessing:'
print d

# Plot the preprocessed first trial
psychic.plots.plot_timeseries(d.nd_xs[0])
plt.savefig('preprocessed.pdf')

# Build classification pipeline
cl = golem.nodes.Chain([
  psychic.nodes.CSP(m=6),
  golem.nodes.FeatMap(var_feat),
  golem.nodes.ModelSelect(
    [golem.nodes.SVM(C=C) for C in np.logspace(-3, 3, 6)],
    lambda d, n: golem.loss.mean_std(golem.loss.accuracy,
      golem.cv.rep_cv(d, n, reps=1, K=5))[0]),
  ])

# Evaluate BCI classifier
test_folds = list(golem.cv.rep_cv(d, cl, reps=5, K=10))
print 'Acc: %.2f (%.2f)' % golem.loss.mean_std(golem.loss.accuracy, test_folds)
