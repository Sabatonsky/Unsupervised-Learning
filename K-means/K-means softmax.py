# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:41:42 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt

D = 2
K = 3
N = 300

X = np.zeros((N, D))
X[:100, :] = np.random.randn(100, D) + (0, 0)
X[100:200, :] = np.random.randn(100, D) + (5, 5)
X[200:300, :] = np.random.randn(100, D) + (0, 5)
Y = np.random.choice(range(K), size = N)

means = np.random.randn(K, D)
new_means = np.zeros((K, D))
new_Y = np.zeros(N)
trial = 0
losses = []

while True:
    new_means[0, :] = X[Y == 0].mean(axis = 0)
    new_means[1, :] = X[Y == 1].mean(axis = 0)
    new_means[2, :] = X[Y == 2].mean(axis = 0)
    
    for i in range(len(new_Y)):
        new_Y[i] = np.argmin(np.linalg.norm(new_means - X[i], axis = 1))
    
    change = np.linalg.norm(new_means - means, ord = 1)
    
    X_loss = np.stack([X,X,X], axis=1)
    dist = X_loss - new_means
    loss = np.min(np.linalg.norm(dist, ord = 1, axis = 2), axis = 1)**2
    losses.append(loss.sum())
    
    trial += 1
    if np.all(Y == new_Y):
        break
    Y = new_Y.copy()
    means = new_means.copy()
        
plt.scatter(X[:,0],X[:,1], c = Y)
plt.scatter(means[:,0], means[:,1], s = 500, c = 'red', marker = '*')
plt.show()
plt.plot(losses)
print(trial)
