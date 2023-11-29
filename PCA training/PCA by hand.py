# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:11:36 2023

@author: Maksim Bannikov
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

array = load_digits()
X = array.data
Y = array.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)
 
covX = np.cov(X_train.T) #Высчитали ковариацию данных самих с собой
lambdas, Q = np.linalg.eigh(covX) #Нашли eigenvalues (1, D) и eigenvectors (D, D).

idx = np.argsort(-lambdas) #Вытаскиваем индексы eigenvalues по убыванию
lambdas = lambdas[idx] #Сортируем eigenvalues
lambdas = np.maximum(lambdas, 0) #Удаляем отрицательные значения
Q = Q[:, idx] #Eigenvectors сортируем аналогичным образом

Z = X_train.dot(Q) #Умножаем X на Eigenvectors, получаем линейную трансформацию, которая меняет исключительно направление осей координат
plt.scatter(Z[:,0], Z[:,1], s=100, c=Y_train, alpha = 0.3)
plt.show()

plt.plot(lambdas)
plt.title("Variance of each component")
plt.show()

plt.plot(np.cumsum(lambdas))
plt.title("Cumulative variance")
plt.show()