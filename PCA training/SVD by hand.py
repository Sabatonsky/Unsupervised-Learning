# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:18:26 2023

@author: Maksim Bannikov
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

array = load_digits()
X = array.data / 255.0
Y = array.target
smoothing = 6.25652e-16 #Smoothing copied from sklearn

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)

N, D = X_train.shape
X_dot = X_train.T.dot(X_train)
lambdas, V = np.linalg.eigh(X_dot)
u_sigma = X_train.dot(V) # By equality
lambdas = np.maximum(lambdas, smoothing) #We derive U through devision on sigma. 
#Zero columns in sigma should be either reduced or smoothed.
sigma = np.sqrt(lambdas)
K = 20 #Reduced dimensionality for X_hat reconstruction

U = u_sigma / sigma

#Matrix sorting for dimensionality reduction
idx = np.argsort(-lambdas)
lambdas = lambdas[idx]
V = V[:, idx]
sigma = sigma[idx]
U = U[:, idx]

#Three ways of linear transformation, including sklearn solution.
#All three are the same.
Z_1 = U.dot(np.diag(sigma))
plt.scatter(Z_1[:,0], Z_1[:,1], s=100, c=Y_train, alpha = 0.3)
plt.show()

SVD = TruncatedSVD()
Z_2 = SVD.fit_transform(X_train)
plt.scatter(Z_2[:,0], Z_2[:,1], s=100, c=Y_train, alpha = 0.3)
plt.show()

Z_3 = X_train.dot(V)
plt.scatter(Z_3[:,0], Z_3[:,1], s=100, c=Y_train, alpha = 0.3)
plt.show()

Z_test = X_test.dot(V)
plt.scatter(Z_test[:,0], Z_test[:,1], s=100, c=Y_test, alpha = 0.3)
plt.show()

U_test = Z_test / sigma

#For further K selection
plt.plot(lambdas)
plt.title("Variance of each component")
plt.show()

plt.plot(np.cumsum(lambdas))
plt.title("Cumulative variance")
plt.show()

#Dimensionality reduction
lambdas = lambdas[:K]
V = V[:, :K]
sigma = sigma[:K]
U = U[:, :K]

#X reconstruction
X_hat = U.dot(np.diag(sigma)).dot(V.T)
X_hat_test = U_test.dot(np.diag(sigma)).dot(V.T)

def compare_random(X_true, X_hat):
    i = np.random.randint(0, len(X_hat))
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(X_true[i].reshape(8,8))
    plt.subplot(2, 1, 2)
    plt.imshow(X_hat[i].reshape(8,8))

#Small tool for reconstruction visual estimation.
#Reconstructed image is more noisy but clearly visible for K = 30.
#Larger K, better quality, lower compression.

compare_random(X_train, X_hat)

#As we can see, only U recalculation is needed for X_test transformation.
compare_random(X_test, X_hat_test)
