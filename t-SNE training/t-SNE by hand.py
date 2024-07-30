# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 12:01:47 2023

@author: Bannikov Maxim

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances

def p_cond(distance_matrix, sigmas):
    smoothing = 10e-20
    P = np.exp(-distance_matrix**2 / (2 * (sigmas **2)))
    np.fill_diagonal(P, smoothing) # Дистанции точек с самими собой нас не интересуют
    P /= np.sum(P, axis = 1) # По формуле P(j | i)
    return P

def p_cond_to_joint(P):
    return (P + P.T) / (2. * P.shape[0]) #Conditional превращаем в joint

def p_joint(X, target_perplexity):
        distance_matrix = pairwise_distances(X, metric='euclidean')
        sigmas = find_optimal_sigmas(distance_matrix, target_perplexity)
        P = p_cond(distance_matrix, sigmas)
        P = p_cond_to_joint(P)
        return P
        
def binary_search(eval_fn, N, target, tol=1e-10, max_iter=10000, lower=10e-20, upper=20000):
    for i in range(max_iter):
        guess = (lower + upper) / 2
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess
        
def calc_perplexity(prob_matrix):
    entropy = - np.sum(prob_matrix*np.log2(prob_matrix), axis = 1)
    perplexity = 2 ** entropy #By perplexity formula
    return perplexity

def perplexity(distance_matrix, sigmas):
    return calc_perplexity(p_cond(distance_matrix, sigmas))

def q_joint(Z):
    smoothing = 10e-20
    distance_matrix = pairwise_distances(Z, metric='euclidean')
    Q = 1 / (1 + distance_matrix**2)
    np.fill_diagonal(Q, smoothing)
    Q /= np.sum(Q) # По формуле Q(i, j)
    return Q, distance_matrix

def find_optimal_sigmas(distance_matrix, target_perplexity):
    N = distance_matrix.shape[0]
    sigmas = np.zeros(N)
    for i in range(N):
        eval_fn = lambda sigma: perplexity(distance_matrix[i:i+1, :], sigma)
        new_sigma = binary_search(eval_fn, N, target_perplexity)
        sigmas[i] = new_sigma
    return sigmas

def tsne_grad(P, Q, Z):
    pq_diff = (P - Q)[:, :, np.newaxis]
    Z_dist = Z[:, np.newaxis, :] - Z #Через np.newaxis заставляем Z подумать, что размерность Z[:, np.newaxis] ниже размерности Z, всвязи с этим numpy расширит Z до размерности N по оси 1, чтобы произвести операцию.
    grad = np.sum(pq_diff * Z_dist, axis=0)
    return grad

def estimate_tsne(X, Y, P, num_iters, learning_rate, momentum, exaggeration):
    N, D = X.shape
    K = 2
    # Инициализируем табличку дистанций, но уже для целевого пространства.
    # В планах через gradient decent оптимизировать до 0 разницу между Q и P. 
    # При этом P у нас фиксированное, а Q мы будем перерасчитывать из Z.
    # Z будем менять на основании функции потерь.
    Z = np.random.normal(0, 0.0001, (N, K))
    Z_2 = Z.copy()
    Z_1 = Z.copy()
    costs = []
    
    for i in range(num_iters):
        if i > 100:
            exaggeration = 1
        Q, distance_matrix = q_joint(Z)
        grad = tsne_grad(exaggeration*P, Q, Z)
        Z += learning_rate * grad
        Z += momentum * (Z_1 - Z_2)
        #Z(N, K) + PQ_diff(N, N, 1) * Z_dist(N, N, K) summed over axis 0 = Z(N,K) + cost(N,K)
        
        Z_2 = Z_1.copy()
        Z_1 = Z.copy()
        
        if i % 100 == 0:
            cost = np.sum(P*np.log(P/Q))
            print("current epoch:", i)
            print("current cost:", cost)
            costs.append(cost)
            plt.scatter(Z[:,0], Z[:,1], c = Y, alpha = 0.5)
            plt.show()
            
    return Z, costs

def main():
    mnist = load_digits()
    X = mnist.data / 255.0
    Y = mnist.target
    target_perplexity = 50
    num_iters = 1000
    learning_rate = 100
    P = p_joint(X, target_perplexity)
    momentum = 0.9
    exaggeration = 12
    
    Z, costs = estimate_tsne(X, Y, P, num_iters, learning_rate, momentum, exaggeration)
    
    plt.plot(costs)
    plt.show()

    plt.scatter(Z[:,0], Z[:,1], c = Y, alpha = 0.5)
    plt.show()
    
if __name__ == "__main__":
  main()