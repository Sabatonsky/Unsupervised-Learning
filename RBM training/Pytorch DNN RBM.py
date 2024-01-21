# -*- coding: utf-8 -*-
"""
Created on Thu Dec    7 21:10:53 2023

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
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
#Hyper parameters
num_features = 784
hidden_sizes = [1000, 750, 500]
num_epochs = 5
pretrain_epochs = 3
batch_size = 500
lr = 10e-4
num_classes = 10

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
        return v1

    def forward_hidden(self, v):
        h = self.hidden(v)
        h = torch.sigmoid(h)
        return h
        
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

class DNN(nn.Module):
    def __init__(self, num_features, hidden_sizes, num_classes):
        super(DNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.linear = nn.Linear(hidden_sizes[-1], num_classes)
        rbm = RBM(num_features, hidden_sizes[0])
        self.hidden_layers.append(rbm)
        for M in range(1, len(hidden_sizes)):
            rbm = RBM(hidden_sizes[M-1], hidden_sizes[M])
            self.hidden_layers.append(rbm)

    def forward(self, x):
        current_input = x
        for rbm in self.hidden_layers:
            z = rbm.forward_hidden(current_input)
            current_input = z
        y = self.linear(current_input)
        return y

model = DNN(num_features, hidden_sizes, num_classes)

#optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#training loop
n_total_steps = len(train_loader)

print("RBM pretraining process")
for epoch in range(pretrain_epochs):
    for i, (images, labels) in enumerate(train_loader):
        current_input = images.reshape(-1, 28*28).to(device)
        for j, rbm in enumerate(model.hidden_layers):
            #forward pretraining
            outputs = rbm.forward(current_input)
            loss = torch.mean(rbm.free_energy(current_input)) - torch.mean(rbm.free_energy(outputs))
            #backwards pretraining
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            current_input = rbm.forward_hidden(current_input)
            if (i+1) % 30 == 0:
                print(f'pretraining RBM layer ({outputs.shape[1]}:{current_input.shape[1]})')
                print(f'pretraining epoch {epoch+1} / {pretrain_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_list = []
                
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #flatten
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)

    
        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (i+1) % 10 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
    loss_list.append(loss.detach().numpy())      
    
plt.plot(loss_list)
plt.show()
            
#test
with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for images, labels in test_loader:
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    outputs = model(images)

    #value, index
    _, predictions = torch.max(outputs, 1)
    n_samples += labels.shape[0]
    n_correct += (predictions == labels).sum().item()
  acc = 100.0 * n_correct / n_samples
  print(f'Test accuracy = {acc}')