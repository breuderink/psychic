import os, unittest
import numpy as np
from braindrain.bdfreader import BDFReader

def status_to_events(status_array):
  '''
  Use the lowest 16 bits to extract events from the status channel.
  Events are encoded as TTL pulses, no event is indicated with the value 0.
  Returns (events, indices)
  '''
  status = np.asarray(status_array, int) & 0xffff # oh I love Python...
  change_ids = np.nonzero(np.concatenate([[1], np.diff(status)]))[0]
  events = status[change_ids]
  return (events[np.nonzero(events)], change_ids[np.nonzero(events)])


if __name__ == '__main__':
  import pylab
  #unittest.main()
  b = BDFReader(open(os.path.join('data', '20090204T1512_kardelen.bdf'), 'rb'))
  channels = b.read_all()
  eeg = channels[:, :-1]
  eeg = channels[:256 * 60,:]
  status = channels[:, -1]
  events, eindices = status_to_events(channels[:, -1])
  #pylab.plot(eeg - np.mean(eeg, axis=0) + np.arange(eeg.shape[1]) * 100)

  for e in [1, 2, 3, 4, 5, 9]:
    ce = events[events == e]
    ci = eindices[events == e]
    #pylab.scatter(ci, ce, color='r', s=2)
    pylab.plot(ci, ce, '+-')

  pylab.legend(('left', 'right', 'left_miss', 'right_miss', 'freeze', 'pauze'))

  pylab.show()
