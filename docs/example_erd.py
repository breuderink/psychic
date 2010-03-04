import golem, psychic
import pylab # for plotting
import numpy as np # for cov features
from scipy import signal

d = psychic.bdf_dataset('CL.bdf')

fs = psychic.get_samplerate(d)
ds = psychic.cut_segments(d, [(219, 220)], [-4 * fs, 4 * fs]) 
print 'Detected %d training sessions.' % len(ds)

# The first two sessions are imaginary movement, the last two are actual
# movement. We will combine the first two:
d = ds[0] + ds[1]

d = psychic.decimate_rec(d, 4)
fs = psychic.get_samplerate(d)
print fs

# Now we have a snippet of raw EEG. 

#f_high = signal.iirfilter(6, [1./(fs/2)], btype='high')
f_beta = signal.iirfilter(6, [8./(fs/2), 30./(fs/2)])
d = psychic.filtfilt_rec(d, f_beta)

car = psychic.nodes.CAR()
d = car.test(d)

d = psychic.slice(d, {2:'left', 3:'right'}, [0, 3 * fs])

print d
d.save('CL.dat')

def var_feat(x):
  return np.var(x, axis=0)

chain = golem.nodes.Chain(
  [psychic.nodes.CSP(m=6),
  golem.nodes.FeatMap(var_feat),
  golem.nodes.SVM(C=100)])

test_folds = list(golem.cv.rep_cv(d, chain, reps=5, K=10))
print golem.loss.mean_std(golem.loss.accuracy, test_folds)

