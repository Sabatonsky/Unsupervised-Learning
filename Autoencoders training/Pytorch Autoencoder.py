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
num_epochs = 2
batch_size = 100
lr = 0.001

#MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class AutoEncoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(AutoEncoder, self).__init__()
    self.encoder = nn.Linear(input_size, hidden_size)
    self.decoder = nn.Linear(hidden_size, input_size)

  def forward(self, x):
    out = F.relu(self.encoder(x))
    out = self.decoder(out)
    out = torch.sigmoid(out)
    return out

model = AutoEncoder(num_features, num_hidden)

#loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#training loop
n_total_steps = len(train_loader)

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
