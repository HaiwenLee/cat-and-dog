import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size = (3,3), padding = 1)
        self.pool1 = nn.AvgPool2d(kernel_size = (4, 4))
        self.conv2 = nn.Conv2d(8,16,kernel_size = (3,3), padding = 1)
        self.pool2 = nn.AvgPool2d(kernel_size = (8, 8))
        self.conv3 = nn.Conv2d(16, 32, kernel_size = (5,5), stride = 2,padding = 2)
        self.linear1 = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 32)
        self.linear3 = nn.Linear(32, 2)
        #self.linear1 = nn.Linear()
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = F.relu(x)
        x = self.pool2(self.conv2(x))
        x= F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.sigmoid(x)

        return x