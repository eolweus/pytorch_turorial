import torch
from torch._C import TracingState
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix

import pdb

torch.set_printoptions(linewidth=120)

def get_train_set():

    train_set = torchvision.datasets.FashionMNIST(
        root='./data'
        ,train=True
        ,download=True
        ,transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    return train_set


if __name__ == "__main__":

    train_set = get_train_set()

    print(len(train_set))

    print(train_set.targets)
    print(train_set.targets.bincount())

    # Getting a single image
    sample = next(iter(train_set))

    image, label = sample

    print(label)
    print(image.shape)

    plt.imshow(image.squeeze(), cmap="gray")
    # plt.show()

    # Getting a batch
    display_loader = torch.utils.data.DataLoader(train_set
                                                , batch_size=10
                                                , shuffle=True)

    batch = next(iter(display_loader))
    images, labels = batch

    '''
    images.shape = ([10, 1, 28, 28])
    labels.shape = ([10])

    '''

    # Makes a grid with a row size of 10
    grid = torchvision.utils.make_grid(images, nrow=10)

    # print(grid)
    # print(grid.shape)
    # print(type(grid))

    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    # plt.show()
    # print('labels:', labels)

