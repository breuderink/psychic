import numpy as np
from golem import DataSet

class RegFilter:
  '''
  Creates a regression filter, used to substract EOG from EEG channels.
  remove_mask is a boolean array containing True if the channel contains EOG.
  Assumes detrended or high-passed data (mean = 0).
  '''
  def __init__(self, noise_channels):
    self.noise_channels = noise_channels

  def train(self, d):
    '''Train the regression filter using signal and noise channels'''
    mask = np.array([i in self.noise_channels for i in range(d.nfeatures)])
    self.cov = np.cov(d.xs, rowvar=False)
    Cnn = self.cov[mask][:, mask]
    Cny = self.cov[mask][:, ~mask]
    self.b = np.dot(np.linalg.inv(Cnn), Cny)

  def test(self, d):
    '''Apply the regression filter, and remove the noise channels'''
    mask = np.array([i in self.noise_channels for i in range(d.nfeatures)])
    xs = d.xs[:, ~mask] - np.dot(d.xs[:, mask], self.b)
    if d.feat_lab != None:
      feat_lab = [d.feat_lab[fi] for fi in range(d.nfeatures) if not mask[fi]]
    else:
      feat_lab = None
    return DataSet(xs=xs, feat_lab=feat_lab, default=d)
