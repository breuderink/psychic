import numpy as np
from golem import DataSet
from ..utils import sliding_window_indices

class SlidingWindow:
  def __init__(self, win_size, win_step, ref_point=.5):
    self.win_size = win_size
    self.win_step = win_step
    self.ref_frame = int(float(ref_point) * (self.win_size - 1))

  def train(self, d):
    pass

  def test(self, d):
    wsize, wstep, refi = self.win_size, self.win_step, self.ref_frame

    xs, ys, ids = [], [], []
    while d.ninstances >= wsize:
      win, d = d[:wsize], d[wstep:]
      xs.append(win.nd_xs)
      ys.append(win.ys[refi])
      ids.append(win.ids[refi])

    if len(xs) == 0:
      return DataSet(xs=np.zeros((0, wsize * d.nfeatures)), 
        feat_shape=(wsize, d.nfeatures), ys=np.zeros((0, d.nclasses)), 
        ids = np.zeros((0, d.ids.shape[1])), 
      default=d)

    xs = np.asarray(xs)
    feat_shape = xs.shape[1:]
    xs = xs.reshape(xs.shape[0], -1)
    ys = np.asarray(ys)
    ids = np.asarray(ids)

    return DataSet(xs=xs, feat_shape=feat_shape, ys=ys, ids=ids, default=d)

class OnlineSlidingWindow (SlidingWindow):
  def __init__(self, win_size, win_step, ref_point=0.5):
    SlidingWindow.__init__(self, win_size, win_step, ref_point)
    self.buffer = None

  def test(self, d):
    if self.buffer != None:
      self.buffer = self.buffer + d
    else:
      self.buffer = d

    wstep, wsize = self.win_step, self.win_size
    buff_size = self.buffer.ninstances

    cons = buff_size - buff_size % self.win_step
    if cons < self.win_size:
      cons = 0
    d, self.buffer = self.buffer[:cons], \
      self.buffer[max(0, cons-wsize + wstep):]
    return SlidingWindow.test(self, d)
