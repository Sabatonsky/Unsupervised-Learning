# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:50:38 2024

@author: AMD
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#Device config
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#Hyper parameters
num_features = 784
num_hidden = [1000, 700, 500]
num_classes = 10
num_epochs = 10
batch_size = 100
lr = 0.01

#MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(num_features, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.l4 = nn.Linear(hidden_sizes[2], num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = torch.sigmoid(out)
        out = self.l2(out)
        out = torch.sigmoid(out)
        out = self.l3(out)
        out = torch.sigmoid(out)
        out = self.l4(out)
        return out

model = NeuralNet(num_features, num_hidden, num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#training loop
n_total_steps = len(train_loader)
l1_w_change = []
l2_w_change = []
l3_w_change = []
last_w = [0,0,0]
current_w = [0,0,0]

last_w[0] = model.l1.weight.detach().numpy().copy()
last_w[1] = model.l2.weight.detach().numpy().copy()
last_w[2] = model.l3.weight.detach().numpy().copy()
    
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
        
        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')    
            
            current_w[0] = model.l1.weight.detach().numpy().copy()
            current_w[1] = model.l2.weight.detach().numpy().copy()
            current_w[2] = model.l3.weight.detach().numpy().copy()
            
            l1_w_change.append(np.abs(current_w[0] - last_w[0]).mean())
            l2_w_change.append(np.abs(current_w[1] - last_w[1]).mean())
            l3_w_change.append(np.abs(current_w[2] - last_w[2]).mean())
            
            last_w[0] = current_w[0].copy()
            last_w[1] = current_w[1].copy()
            last_w[2] = current_w[2].copy()
      
plt.plot(l1_w_change, label='l1')
plt.plot(l2_w_change, label='l2')
plt.plot(l3_w_change, label='l3')
plt.legend()
plt.show()