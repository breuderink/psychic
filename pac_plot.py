import os, unittest
import numpy as np
import golem
import psychic

if __name__ == '__main__':
  import pylab
  b = psychic.bdfreader.BDFReader(open(os.path.join('data', '20090204T1512_kardelen.bdf'), 'rb'))
  channels = b.read_all()
  eeg = channels[:, :-1]
  eeg = channels[:256 * 60,:]
  status = channels[:, -1]
  events, eindices = psychic.preprocessing.status_to_events(channels[:, -1])
  #pylab.plot(eeg - np.mean(eeg, axis=0) + np.arange(eeg.shape[1]) * 100)

  for e in [1, 2, 3, 4, 5, 9]:
    ce = events[events == e]
    ci = eindices[events == e]
    #pylab.scatter(ci, ce, color='r', s=2)
    pylab.plot(ci, ce, '+-')

  pylab.legend(('left', 'right', 'left_miss', 'right_miss', 'freeze', 'pauze'))

  pylab.show()
