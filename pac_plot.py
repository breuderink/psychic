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

class TestEventDetection(unittest.TestCase):
  def setUp(self):
    pass

  def do_test(self, status, events, indices):
    e, i = status_to_events(status)
    np.testing.assert_equal(e, events)
    np.testing.assert_equal(i, indices)

  def test_single_event(self):
    self.do_test([0, 0, 0, 1, 1, 0, 0, 0], [1], [3])

  def test_functionality(self):
    self.do_test([2, 0, 1, 1, 3, 0, 4], [2, 1, 3, 4], [0, 2, 4, 6])

  def test_degenerate(self):
    self.do_test([0], [], [])

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
