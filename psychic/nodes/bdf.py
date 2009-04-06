import numpy as np
from golem import DataSet
from ..bdfreader import BDFReader

STATUS = 'Status'

class BDFFile:
  def __init__(self, fname):
    self.fname = fname

  def train(self, d=None):
    return self.test(d)

  def test(self, d=None):
    if d != None:
      assert d.ninstances == 0, 'This node only *generates* data.'
    f = open(self.fname, 'rb')
    try:
      b = BDFReader(f)
      frames = b.read_all()

      data_mask = [i for i, lab in enumerate(b.labels) if lab != STATUS]
      status_mask = b.labels.index(STATUS)
      feat_lab = [b.labels[i] for i in data_mask]
      assert min(b.sample_rate) == max(b.sample_rate)
      sample_rate = b.sample_rate[0]
      ids = (np.arange(frames.shape[0]) / float(sample_rate)).reshape(-1, 1)

      print data_mask, frames.shape
      d = DataSet(
        xs=frames[:,data_mask], 
        ys=frames[:,status_mask].reshape(-1, 1), 
        ids=ids, feat_lab=feat_lab, cl_lab=['status'], 
        extra={'sample_rate': b.sample_rate})
    finally:
      f.close()
    return d
