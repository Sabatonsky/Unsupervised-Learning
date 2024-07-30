# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper parameters
num_features = 784
num_hidden = 64
num_epochs = 3
batch_size = 100
lr = 0.001
pretrain_epochs = 2

#MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class AutoEncoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(AutoEncoder, self).__init__()
    self.encoder = nn.Linear(input_size, hidden_size)
    self.bias_o = nn.Parameter(torch.zeros(input_size))
    self.input_size = input_size
    self.hidden_size = hidden_size

  def forward(self, x):
    out = F.relu(self.encoder(x))
    out = F.linear(out, weight = self.encoder.weight.transpose(0,1), bias = self.bias_o)
    out = torch.sigmoid(out)
    return out

  def encode(self, x):
    out = F.relu(self.encoder(x))
    return out

  def decode(self, x):
    out = F.linear(x, weight = self.encoder.weight.transpose(0,1), bias = self.bias_o)
    out = torch.sigmoid(out)
    return out

class DNN(nn.Module):
  def __init__(self, input_size, hidden_sizes, UnsupervisedModel = AutoEncoder):
    super(DNN, self).__init__()
    self.layers = nn.ModuleList()
    current_input = input_size
    for hidden_size in hidden_sizes:
      ae = UnsupervisedModel(current_input, hidden_size)
      self.layers.append(ae)
      current_input = hidden_size

  def forward(self, X):
    current_input = X
    for ae in self.layers:
      Z = ae.encode(current_input)
      current_input = Z
    for ae in reversed(self.layers):
      Z = ae.decode(current_input)
      current_input = Z
    return Z

model = DNN(num_features, [750, 500, 250])
print(model.layers)

#loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#training loop
n_total_steps = len(train_loader)

#pretrain cycle
for epoch in range(pretrain_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.reshape(-1, 28*28).to(device)
    current_input = images
    #print(current_input.shape)
    ae = model.layers[0]
    #print(f'Curren layer: [{ae.input_size}:{ae.hidden_size}]')
    outputs = ae(current_input)
    loss = criterion(current_input, outputs)
    current_input = outputs
    #backwards
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
      print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

#final tuning
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    #flatten
    images = images.reshape(-1, 28*28).to(device)

    #forward
    outputs = model(images)
    loss = criterion(outputs, images)

    #backwards
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
      print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

#test
with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for images, _ in test_loader:
    images = images.reshape(-1, 28*28).to(device)
    outputs = model(images)

  for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i].reshape(28, 28), cmap='gray')
    plt.savefig('original images')
  for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(outputs[i].reshape(28, 28), cmap='gray')
    plt.savefig('restored images')
