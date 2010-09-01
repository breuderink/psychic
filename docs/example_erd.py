import logging, glob
import golem, psychic
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def logvar_feat(x):
  return np.log(np.var(x, axis=0))

def window(x):
  return x * np.hanning(x.shape[0]).reshape(-1, 1)

# Define preprocessing pipeline
preprocessing = golem.nodes.Chain([
  # Clip extreme values
  psychic.nodes.Winsorize(),
  # Filter to beta range (8--30 Hz)
  psychic.nodes.Filter(lambda s : signal.iirfilter(6, [8./(s/2), 30./(s/2)])),
  # Extract 2 second window centered on key-press
  psychic.nodes.Slice({1:'left', 2:'right'}, [-.7, .7]),
  ])

pipeline = golem.nodes.Chain([
  golem.nodes.FeatMap(window),
  psychic.nodes.CSP(m=6),
  golem.nodes.FeatMap(logvar_feat),
  golem.nodes.SVM(C=1e5)])

# Setup logging levels
logging.basicConfig(level=logging.WARNING)
logging.getLogger('golem.nodes.ModelSelect').setLevel(logging.INFO)

# Load dataset (see also psychic.helpers.bdf_dataset for .bdf files)
d = golem.DataSet.load('S9.dat')
d0 = d

# Preprocess
preprocessing.train(d) # Required to calculate sampling rate
d = preprocessing.apply(d)

NTRAIN = 500
d, dtest = d[:NTRAIN], d[NTRAIN:]

# Build classifier
pipeline.train(d)

# Do predictions
pred = pipeline.apply(dtest)

print 'Acc:', golem.perf.accuracy(pred)
print 'AUC:', golem.perf.auc(pred)

