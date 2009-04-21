from golem import DataSet

class ChannVar:
  def train(self, d):
    pass

  def test(self, d):
    assert len(d.feat_shape) > 1
    xs = d.nd_xs.var(axis=1).reshape(d.ninstances, -1)
    return DataSet(xs=xs, default=d)
