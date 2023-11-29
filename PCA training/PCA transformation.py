# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:07:46 2023

@author: Maksim Bannikov
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

array = load_digits()
X = array.data
Y = array.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)

pca = PCA()
reduced = pca.fit_transform(X_train)
plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Y_train, alpha = 0.5)
plt.show()

plt.plot(pca.explained_variance_ratio_)
plt.show()

cumulative = []
last = 0

for v in pca.explained_variance_ratio_:
    cumulative.append(last + v)
    last = cumulative[-1]
plt.plot(cumulative)
plt.show()

