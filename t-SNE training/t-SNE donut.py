# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:29:54 2023

@author: AMD
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_donut():
  N = 600
  R_inner = 10
  R_outer = 20
  R1 = np.random.randn(int(N/2)) + R_inner
  theta = 2*np.pi*np.random.random(int(N/2))
  X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

  R2 = np.random.randn(int(N/2)) + R_outer
  theta = 2*np.pi*np.random.random(int(N/2))
  X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

  X = np.concatenate([X_inner, X_outer])
  Y = np.array([0]*(int(N/2)) + [1]*(int(N/2)))
  return X, Y

def main():
    X, Y = get_donut()
    model = TSNE(perplexity=30)
    Z = model.fit_transform(X)
    
    plt.scatter(X[:,0], X[:,1], s=100, c=Y)
    plt.show()
    
    plt.scatter(Z[:,0], Z[:,1], s=100, c=Y)
    plt.show()

if __name__ == "__main__":
    main()