# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:52:40 2023

@author: AMD
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

mnist = load_digits()
X = mnist.data / 255.0
Y = mnist.target

model = TSNE(perplexity=10, learning_rate = 10e2, init="random")
Z = model.fit_transform(X)

plt.scatter(Z[:,0], Z[:,1], s=100, c=Y, alpha = 0.5)
plt.show()
