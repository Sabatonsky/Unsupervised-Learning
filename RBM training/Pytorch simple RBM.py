# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:10:53 2023

@author: Maksim Bannikov
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Hyper parameters
num_features = 784
num_epochs = 20
batch_size = 500
lr = 10e-3

#MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class RBM(nn.Module):
  def __init__(self, input_size, hidden_size, k=1):
    super(RBM, self).__init__()
    self.hidden = nn.Linear(input_size, hidden_size).to(device)
    self.bias_b = nn.Parameter(torch.zeros(input_size).to(device))
    self.k = k

  def forward(self, v):
    h1 = self.sample_h_given_v(v)
    h = h1
    for _ in range(self.k):
      v1 = self.sample_v_given_h(h)
      h = self.sample_h_given_v(v1)
    return v, v1

  def sample_h_given_v(self, v):
    out = F.linear(v, weight = self.hidden.weight, bias = self.hidden.bias)
    p_h_given_v = torch.sigmoid(out)
    h_sample = p_h_given_v.bernoulli()
    return h_sample.to(device)

  def sample_v_given_h(self, v):
    out = F.linear(v, weight = self.hidden.weight.transpose(0,1), bias = self.bias_b)
    p_v_given_h = torch.sigmoid(out)
    v_sample = p_v_given_h.bernoulli()
    return v_sample.to(device)

  def free_energy(self, v):
    return -v.matmul(self.bias_b) - torch.sum(torch.log(1+torch.exp(v.matmul(self.hidden.weight.transpose(0,1)) + self.hidden.bias)), axis = 1)

num_hidden = 400
model = RBM(num_features, num_hidden)

#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    #flatten
    images = images.reshape(-1, 28*28).to(device)
    #forward
    v, v1 = model.forward(images)
    loss = torch.mean(model.free_energy(v)) - torch.mean(model.free_energy(v1))

    #backwards
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
      print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
      
#test
with torch.no_grad():
  v = v.to('cpu')
  v1 = v1.to('cpu')

  for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(v[i].reshape(28, 28), cmap='gray')
  plt.show()
    
  for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(v1[i].reshape(28, 28), cmap='gray')
  plt.show()
