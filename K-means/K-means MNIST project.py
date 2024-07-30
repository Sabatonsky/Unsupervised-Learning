# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
import pandas as pd

def d(u, v):
  return np.linalg.norm(v - u, ord = 2, axis = 1)

def cost(X, R, M):
  cost = 0
  for k in range(len(M)):
    cost += np.diag(R[:, k]).dot(d(M[k], X)).sum()
  return cost

def plot_k_means(X, Y, K, max_iters=20, beta=1.0):
  N, D = X.shape
  M = np.zeros((K, D))
  R = np.zeros((N, K))

  for k in range(K):
    M[k] = X[np.random.choice(N)]
    costs = np.zeros(max_iters)
    purities = np.zeros(max_iters)
    dbis = np.zeros(max_iters)

  for i in range(max_iters):

    for k in range(K):
      num = np.exp(-beta*d(M[k], X))
      den = np.sum(np.exp(-beta*d(M[j], X)) for j in range(K))**-1
      R[:, k] = np.diag(num).dot(den)
    for k in range(K):
      M[k] = R[:, k].dot(X) / R[:, k].sum()
    costs[i] = cost(X, R, M)
    purities[i] = purity(Y, R)
    dbis[i] = DBI(X, M, R)

    if i > 0:
      if np.abs(costs[i] - costs[i-1]) < 10e-8:
        break

  plt.plot(costs)
  plt.title('Costs')
  plt.savefig('fig_1')
  plt.clf()

  plt.plot(purities)
  plt.title('Purity')
  plt.savefig('fig_2')
  plt.clf()

  plt.plot(dbis)
  plt.title('DBI')
  plt.savefig('fig_3')
  plt.clf()

def main():
  mnist = load_digits()

  df = pd.DataFrame(mnist.data)
  X = df.to_numpy(dtype='float32') / 255.0
  Y = pd.DataFrame(mnist.target).to_numpy().flatten()
  X, Y = shuffle(X, Y)

  N, D = X.shape
  K = len(set(Y))
  plot_k_means(X, Y, K, max_iters = 30, beta = 100)

def purity(Y, R):
  purity = 0
  N, K = R.shape
  for k in range(K):
    max_itst = 0
    for j in range(K):
      itst = R[Y==j, k].sum()
      if itst > max_itst:
        max_itst = itst
    purity += max_itst
  return purity/N

def DBI(X, M, R):
  K, D = M.shape
  sigma = np.zeros(K)

  for k in range(K):
    diff = X - M[k]
    sqr_dist = (diff*diff).sum(axis = 1)
    wsd = R[:, k]*sqr_dist
    sigma[k] = np.sqrt(wsd).mean()

  dbi = 0

  for k in range(K):
    max_ratio = 0
    for j in range(K):
      if k != j:
        num = sigma[k] + sigma[j]
        den = np.linalg.norm(M[k] - M[j])
        ratio = num / den
        if max_ratio < ratio:
          max_ratio = ratio
    dbi += max_ratio
  return dbi / K

if __name__ == '__main__':
  main()
