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

def donut():
  N = 1000
  R_inner = 5
  R_outer = 15
  R1 = np.random.randn(int(N/2)) + R_inner
  theta = 2*np.pi*np.random.random(int(N/2))
  X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
  R2 = np.random.randn(int(N/2)) + R_outer
  X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

  X = np.concatenate([X_inner, X_outer])
  return X

def main():
  X = donut()
  plot_k_means(X, 2)

  X = np.zeros((1000, 2))
  X[:500, :] = np.random.multivariate_normal([0,0], [[1,0], [0,20]], 500)
  X[500:, :] = np.random.multivariate_normal([5,0], [[1,0], [0,20]], 500)
  plot_k_means(X, 2)

  X = np.zeros((1000, 2))
  X[:950, :] = np.array([0,0]) + np.random.randn(950, 2)
  X[950:, :] = np.array([5,0]) + np.random.randn(50, 2)
  plot_k_means(X, 2)

if __name__ == '__main__':
  main()
