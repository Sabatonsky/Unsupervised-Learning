# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
import matplotlib.pyplot as plt

def d(u, v):
  return np.linalg.norm(v - u, ord = 2, axis = 1)

def cost(X, R, M):
  cost = 0
  for k in range(len(M)):
    cost += np.diag(R[:, k]).dot(d(M[k], X)).sum()
  return cost

def plot_k_means(X, K, max_iters=20, beta=1.0):
  N, D = X.shape
  M = np.zeros((K, D))
  R = np.zeros((N, K))

  for k in range(K):
    M[k] = X[np.random.choice(N)]
    costs = np.zeros(max_iters)

  grid_width = 5
  grid_height = int(max_iters / grid_width)
  random_colors = np.random.random((K, 3))
  plt.figure()

  for i in range(max_iters):
    colors = R.dot(random_colors)
    plt.subplot(grid_width, grid_height, i+1)
    plt.scatter(X[:,0], X[:,1], c = colors)

    for k in range(K):
      num = np.exp(-beta*d(M[k], X))
      den = np.sum(np.exp(-beta*d(M[j], X)) for j in range(K))**-1
      R[:, k] = np.diag(num).dot(den)

    for k in range(K):
      M[k] = R[:,k].dot(X) / R[:,k].sum()

    costs[i] = cost(X, R, M)

    if i > 0:
      if np.abs(costs[i] - costs[i-1]) < 0.1:
        break

  plt.savefig('fig_2')
  plt.clf()

  plt.plot(costs)
  plt.title('Costs')
  plt.savefig('fig_1')
  plt.clf()

def main():
  D = 2
  s = 4
  mu1 = np.array([0, 0])
  mu2 = np.array([s, s])
  mu3 = np.array([0, s])
  N = 900
  X = np.zeros((N, D))
  X[:300, :] = np.random.randn(300, D) + mu1
  X[300:600, :] = np.random.randn(300, D) + mu2
  X[600:900, :] = np.random.randn(300, D) + mu3

  K = 3
  plot_k_means(X, K, max_iters = 30, beta = 1)

if __name__ == '__main__':
  main()
