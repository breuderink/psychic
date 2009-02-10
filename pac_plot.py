import os, unittest, logging
import numpy as np
import pylab
import golem
import psychic
from psychic.preprocessing import *

if __name__ == '__main__':
  logging.basicConfig()
  b = psychic.bdfreader.BDFReader(open(os.path.join('data', '20090204T1512_kardelen.bdf'), 'rb'))
  channels = b.read_all()
  eeg = channels[:, :-1]
  #eeg = channels[:256 * 60,:]
  status = channels[:, -1]
  events, eindices = psychic.preprocessing.status_to_events(channels[:, -1])
  #pylab.plot(eeg - np.mean(eeg, axis=0) + np.arange(eeg.shape[1]) * 100)

  #for e in [1, 2, 3, 4, 5, 9]:
  #  ce = events[events == e]
  #  ci = eindices[events == e]
  #  #pylab.scatter(ci, ce, color='r', s=2)
  #  pylab.plot(ci, ce, '+-')

  #pylab.legend(('left', 'right', 'left_miss', 'right_miss', 'freeze', 'pauze'))


  car_eeg = car(eeg)
  mean_left = slice(car_eeg, eindices[events==1], pre_frames=256, post_frames=256).mean(axis=0)
  mean_right = slice(car_eeg, eindices[events==2], pre_frames=256, post_frames=256).mean(axis=0)
  pylab.subplot(211)
  psychic.plots.plot_timeseries(mean_right, 5)
  pylab.axvline(256, color='r', linewidth=2)
  pylab.xticks(np.linspace(0, 512, 5), 
    ['%1.2fs' % i for i in np.linspace(-1, 1, 5)])
  pylab.title('right')

  pylab.subplot(212)
  psychic.plots.plot_timeseries(mean_left, 5)
  pylab.axvline(256, color='r', linewidth=2)
  pylab.xticks(np.linspace(0, 512, 5), 
    ['%1.2fs' % i for i in np.linspace(-1, 1, 5)])
  pylab.title('left')
  pylab.show()
