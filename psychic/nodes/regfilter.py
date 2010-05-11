import numpy as np
from golem import DataSet
from golem.nodes import BaseNode

class RegFilter(BaseNode):
  '''
  Creates a regression filter, used to substract EOG from EEG channels.
  remove_mask is a boolean array containing True if the channel contains EOG.
  Assumes detrended or high-passed data (mean = 0).

  See [1].

  [1] Alois Schloegl, Claudia Keinrath, Doris Zimmermann, Reinhold Scherer,
  Robert Leeb, and Gert Pfurtscheller. A fully automated correction method of
  EOG artifacts in EEG recordings. Clinical Neurophysiology, 118:98--104, 2007.
  '''
  def __init__(self, noise_channels):
    BaseNode.__init__(self)
    self.noise_channels = noise_channels

  def train_(self, d):
    '''Train the regression filter using signal and noise channels'''
    mask = np.array([i in self.noise_channels for i in range(d.nfeatures)])
    self.cov = np.cov(d.xs, rowvar=False)
    Cnn = self.cov[mask][:, mask]
    Cny = self.cov[mask][:, ~mask]
    self.b = np.dot(np.linalg.inv(Cnn), Cny)

  def apply_(self, d):
    '''Apply the regression filter, and remove the noise channels'''
    mask = np.array([i in self.noise_channels for i in range(d.nfeatures)])
    xs = d.xs[:, ~mask] - np.dot(d.xs[:, mask], self.b)
    if d.feat_lab != None:
      feat_lab = [d.feat_lab[fi] for fi in range(d.nfeatures) if not mask[fi]]
    else:
      feat_lab = None
    return DataSet(xs=xs, feat_lab=feat_lab, default=d)
