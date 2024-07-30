# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:37:24 2023

@author: Bannikov Maxim
"""

import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.utils import shuffle

code = ['A', 'T', 'C', 'G']

def to_code(a):
    return [code[i] for i in a]

def dist(a, b):
    return sum(i != j for i, j in zip(a, b))
    
def generate_offspring(parent):
    return [mutate(c) for c in parent]

def mutate(c):
    if np.random.random() < 0.001:
        return np.random.choice(code)
    return c

parent = []

for _ in range(3):
    p_i = to_code(np.random.randint(4, size = 1000))
    parent.append(p_i)
    
epochs = 99
max_offspring = 1000

for i in range(epochs):
    next_generation = []
    for p in parent:
        num_offspring = np.random.randint(3) + 1
        
        for _ in range(num_offspring):
            child = generate_offspring(p)
            next_generation.append(child)
    
    parent = next_generation
    
    parent = shuffle(parent)
    parent = parent[:max_offspring]
    
    print("Gen %d / %d, size = %d" % (i + 1, epochs, len(parent)))
    
N = len(parent)
dist_matrix = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        elif i < j:
            a = parent[i]
            b = parent[j]
            dist_matrix[i, j] = dist(a, b)
        else:
            dist_matrix[i, j] = dist_matrix[j, i]
    if i % 10 == 0:
        print("current i:", i)

dist_array = ssd.squareform(dist_matrix)

Z = linkage(dist_array, 'ward')
plt.title("Ward")
dendrogram(Z, color_threshold=8000)
plt.show()
