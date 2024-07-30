# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def E(X, pi, mu, cov):
  N, D = X.shape
  K = len(pi)

def d(u, v):
  return np.linalg.norm(v - u, ord = 2, axis = 1)

def plot_k_means(X, K, max_iters=20, smoothing=1e-2):
  N, D = X.shape
  mu = np.zeros((K, D))
  pi = np.ones(K) / K
  cov = np.zeros((K, D, D))

  for k in range(K):
    mu[k] = X[np.random.choice(N)]
    cov[k] = np.eye(D)
    costs = np.zeros(max_iters)

  grid_width = 5
  grid_height = int(max_iters / grid_width)
  random_colors = np.random.random((K, 3))
  plt.figure()

  lls = []
  gamma = np.zeros((N, K))

  for i in range(max_iters):
    for k in range(K):
      gamma[:, k] = pi[k]*multivariate_normal.pdf(X, mu[k], cov[k])
    R = gamma / gamma.sum(axis=1, keepdims=True)
    colors = R.dot(random_colors)
    plt.subplot(grid_width, grid_height, i+1)
    plt.scatter(X[:,0], X[:,1], c = colors)
    N_k = np.sum(R, axis = 0)

    for k in range(K):
      pi[k] = N_k[k] / N
      mu[k, :] = R[:, k].dot(X) / N_k[k]
      dist = X - mu[k]
      num = (np.expand_dims(R[:, k], -1)*dist).T.dot(dist)
      cov[k] = num / N_k[k] + np.eye(D)*smoothing

    ll = np.log(gamma.sum(axis=1)).sum()
    lls.append(ll)

    if i > 0:
      if np.abs(lls[i] - lls[i-1]) < 0.1:
        break

  plt.savefig('fig_2')
  plt.clf()

  plt.plot(lls)
  plt.title('Log-Likelihood')
  plt.savefig('fig_1')
  plt.clf()

def main():
  D = 2
  s = 4
  mu1 = np.array([0, 0])
  mu2 = np.array([s, s])
  mu3 = np.array([0, s])
  N = 2000
  X = np.zeros((N, D))
  X[:1200, :] = np.random.randn(1200, D)*2 + mu1
  X[1200:1800, :] = np.random.randn(600, D) + mu2
  X[1800:, :] = np.random.randn(200, D)*0.5 + mu3

  K = 3
  plot_k_means(X, K, max_iters = 30)

if __name__ == '__main__':
  main()
