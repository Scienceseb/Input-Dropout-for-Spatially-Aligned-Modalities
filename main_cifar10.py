import sys
import torch
import torchvision
from torchvision.datasets import ImageFolder
import os
from shutil import copyfile
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
import torch.utils.data as data
from torchvision import datasets, models, transforms
from training import train
from training import test
import numpy as np
import time
from torch.utils.data.sampler import SubsetRandomSampler
import re
import torch
from collections import OrderedDict
import random
from PIL import Image
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class InputDropout(object):
    def __init__(self, mode="addit",test=False):
        self.mode = mode
        self.test=test

    def __call__(self, sample):
        if self.test == True:
            data[3, :, :] = 0
        if self.test==False:
            if self.mode == "both":
                if random.uniform(0, 1) <= 0.66:
                    if random.uniform(0, 1) <= 0.5:
                        data[3, :, :] = 0
                    else:
                        data[0, :, :] = 0
                        data[1, :, :] = 0
                        data[2, :, :] = 0

            elif random.uniform(0, 1) <= 0.5:
                if self.mode == "addit":
                    data[3, :, :] = 0

        return data



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4*32*32, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 4*32*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def CIFAR_dataset(train_split_percent=0.8, mode="addit"):
    transform_train = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)), InputDropout(mode)])
    transform_val_test = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)), InputDropout(test)])

    test_set = datasets.ImageFolder('path_to_testset',transform=transform_val_test) #200 images are selected as test set
    train_set = datasets.ImageFolder('path_to_trainset',transform=transform_train)  #the rest of the images serve as a training set.
    val_set = datasets.ImageFolder('path_to_trainset',transform=transform_val_test) #200 images are selected as validation set

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor((1 - train_split_percent) * num_train))
    np.random.seed(1)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, sampler=train_sampler, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=64, sampler=valid_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, num_workers=8)

    return train_loader, valid_loader, test_loader





if __name__ == '__main__':

    for i in range(0,10):


        #Train with the BW always to 0
        model = Net()
        model.type(torch.cuda.FloatTensor)
        train_loader, valid_loader, test_loader = pixar_dataset(0.8,'addit', test=True) #both work also, because of test=true, the mode option is useless here

        use_gpu = True
        n_epoch = 100
        learning_rate = 0.002

        history = train(model, train_loader, valid_loader, n_epoch, learning_rate, use_gpu=use_gpu)
        history.display()

        test_acc, test_loss = test(model, test_loader, use_gpu=use_gpu)
        print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))




        #Train with Input Dropout in addit mode
        model = Net()
        model.type(torch.cuda.FloatTensor)
        train_loader, valid_loader, test_loader = pixar_dataset(0.8, 'addit',test=False)


        use_gpu = True
        n_epoch = 100
        learning_rate = 0.002

        history = train(model, train_loader, valid_loader, n_epoch, learning_rate, use_gpu=use_gpu)
        history.display()

        test_acc, test_loss = test(model, test_loader, use_gpu=use_gpu)
        print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))


        #Train with Input Dropout in both mode
        model = Net()
        model.type(torch.cuda.FloatTensor)
        train_loader, valid_loader, test_loader = pixar_dataset(0.8, 'both', test=False)

        use_gpu = True
        n_epoch = 100
        learning_rate = 0.002

        history = train(model, train_loader, valid_loader, n_epoch, learning_rate, use_gpu=use_gpu)
        history.display()

        test_acc, test_loss = test(model, test_loader, use_gpu=use_gpu)
        print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))









