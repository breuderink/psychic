import operator, logging
import numpy as np
# TODO: 
# [ ] Add NaN handling to enable working with missing data

def ribs(loadings):
  '''
  Convert a list of n loading matrices [A_{fi}, B_{fj}, C_{fk}, ...] into ribs
  [A_{fi11...}, B_{f1j1...}, C_{f11k...}, ...]. These ribs can be multiplied
  with numpy broadcasting to fill a tensor with data.
  '''
  loadings = [np.atleast_2d(l) for l in loadings]
  nfactors = loadings[0].shape[0]
  assert np.alltrue([l.ndim == 2 and l.shape[0] == nfactors for l in loadings])
  ribs = []
  for mi in range(len(loadings)):
    shape = [nfactors] + [-1 if fi == mi else 1 for fi in range(len(loadings))]
    ribs.append(loadings[mi].reshape(shape))
  return ribs

def para_compose(ribs):
  return np.sum(reduce(operator.mul, ribs), axis=0)

def parafac(x, nfactors=3, max_iter=5000):
  '''
  PARAFAC is a multi-way tensor decomposition method. Given a tensor X, and a
  number of factors nfactors, PARAFAC decomposes the X in n factors for each 
  dimension in X using alternating least squares.

  PARAFAC can be seen as a generalization of PCA to higher order arrays [1].
  Return a list with a [nfactors x mi] arrays, one for each mode.

  [1] Rasmus Bro. PARAFAC. Tutorial and applications. Chemometrics and
  Intelligent Laboratory Systems, 38(2):149-171, 1997.
  '''
  log = logging.getLogger('psychic.parafac')
  loadings = [np.random.rand(nfactors, n) for n in x.shape]
  
  last_mse = np.inf
  for i in range(max_iter):
    # 1) forward (predict x)
    xhat = para_compose(ribs(loadings))

    # 2) stopping?
    mse = np.mean((xhat - x) ** 2)
    log.info('MSE:%.2g' % mse)
    if last_mse - mse < 1e-10 or mse < 1e-20:
      break
    last_mse = mse

    for mode in range(len(loadings)):
      log.debug('iter: %d, dir: %d' % (i, mode))
      # a) Re-compose using other factors
      Z = ribs([l for li, l in enumerate(loadings) if li != mode])
      Z = reduce(operator.mul, Z)

      # b) Isolate mode
      #Z = Z.reshape(-1, nfactors) # Z = [long x fact]
      Z = Z.reshape(nfactors, -1).T # Z = [long x fact]
      Y = np.rollaxis(x, mode)
      Y = Y.reshape(Y.shape[0], -1).T # Y = [mode x long]

      # c) least squares estimation: x = np.lstsq(Z, Y) -> Z x = Y
      new_fact, _, _, _ = np.linalg.lstsq(Z, Y)
      loadings[mode] = new_fact
  return loadings

def normalized_loadings(loadings):
  '''
  Normalize the PARAFAC loadings by normalizing each rib to unit length.
  Factors are sorted by combined magnitude s:

  x_{ijk..} = \sum a_{if} b_{jf} c_{kf} ... = 
    \sum s_f * a_{if} b_{jf} c_{kf) ..

  Return a tuple (magnitudes, norm_loadings)
  '''
  mags = np.asarray([np.apply_along_axis(np.linalg.norm, 1, mode)
    for mode in loadings])

  norm_loadings = [loadings[mi] / mags[mi].reshape(-1, 1) 
    for mi in range(len(loadings))]

  mags = np.prod(mags, axis=0)
  order = np.argsort(mags)[::-1]
 
  return mags[order], [np.asarray([mode[fi] for fi in order])
    for mode in norm_loadings]
