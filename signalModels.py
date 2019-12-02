import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random,time

class seqCNN(nn.Module):
    def __init__(self,input_channels,output_size,kernel_sizes):
        """For doc string"""
        super(seqCNN,self).__init__()
        

        """
        -------------------------------------------------------------
                            PARAMETER DEFINITIONS

        define some variables for readability and maintainability 
        -------------------------------------------------------------
        """
        #kernel sizes
        k1_size = kernel_sizes[0]
        k2_size = kernel_sizes[1]
        k3_size = kernel_sizes[2]

        f1_num_filters = 8
        f2_num_filters = 2*f1_num_filters
        f3_num_filters = 2*f2_num_filters

        fc1_num_neurons = 352
        fc2_num_neurons = 120
        fc3_num_neurons = 60

        max_pool_kernel = 3
        number_of_pools = 2 #how many times we perform the pooling

        """
        -------------------------------------------------------------
                            LAYER DEFINITIONS
        -------------------------------------------------------------
        """
        #filter (fi)/convolutional layers
        self.f1 = nn.Conv1d(input_channels, f1_num_filters, k1_size)
        self.bn_f1 = nn.BatchNorm1d(f1_num_filters)

        self.f2 = nn.Conv1d(f1_num_filters, f2_num_filters, k2_size)
        self.bn_f2 = nn.BatchNorm1d(f2_num_filters)

        self.f3 = nn.Conv1d(f2_num_filters, f3_num_filters, k3_size)
        

        self.pool = nn.MaxPool1d(max_pool_kernel)

        #fully connected layers
        self.fc1 = nn.Linear(fc1_num_neurons, fc2_num_neurons)
        self.fc2 = nn.Linear(fc2_num_neurons, fc3_num_neurons)
        self.fc3 = nn.Linear(fc3_num_neurons, output_size)
        
    def forward(self,X):
        out = F.relu(self.bn_f1(self.f1(X)))
        out = F.relu(self.bn_f2(self.f2(out)))
        out = self.pool(out)
        out = F.relu(self.f3(out))
        out = self.pool(out)
        out = out.view(-1,out.shape[1]*out.shape[2])
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out