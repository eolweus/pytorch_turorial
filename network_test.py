import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix
from resources.runBuilder import RunBuilder

torch.set_printoptions(linewidth=120)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # the first in_channel depends on the number of color channels present in our dataset
        # for gray scale images, like the ones in the mnist dataset, that is one.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        # The 12 in in_features comes from the 12 in out_channels above
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        # out_features is equal to the number of classes in our training set
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)

        # We won't use softmax here because we are going to use the cross_entropy function
        # t = F.softmax(t, dim=1)

        return t


def get_train_set():

    # We're using the FashinMNIST dataset
    train_set = torchvision.datasets.FashionMNIST(
        root='./data'
        ,train=True
        ,download=True
        ,transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    return train_set


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# This decorater makes sure we don't unnecessarily keep track of grads when we don't need to
@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )


if __name__ == "__main__":
    # torch.set_grad_enabled(False)
    
    network = Network()

    train_set = get_train_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    optimizer = optim.Adam(network.parameters(), lr=0.01)

    for epoch in range(1):

        total_correct = 0
        total_loss = 0

        for batch in train_loader: # Get batch

            images, labels = batch

            preds = network(images) # Pass batch
            loss = F.cross_entropy(preds, labels) # Calcualte loss
            
            optimizer.zero_grad() # Zeros out the gradients before the next calcualtion
            loss.backward() # Calculate Gradients
            optimizer.step() # Update Weights

            total_correct += get_num_correct(preds, labels)
            total_loss += loss.item()
        
        print("epoch:", epoch, "total_correct:", total_correct, "loss:", total_loss)
        print("accuracy:", total_correct / len(train_set))
    
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=1000)
    train_preds = get_all_preds(network, prediction_loader)



    # print(preds)
    # print(F.softmax(preds, dim=1))
    # print(labels)
    # print(preds.argmax(dim=1))
    # print("correct predictions:", get_num_correct(preds, labels))