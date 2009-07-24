import numpy as np
from golem.loss import accuracy

def wolpaw_bitr(N, P):
  assert 0 <= P <= 1
  assert 2 <= N
  result = np.log2(N)
  if P > 0: 
    result += P * np.log2(P)
  if P < 1:
    result += (1 - P) * np.log2((1 - P)/(N - 1.))
  return result
