import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from Bio import SeqIO
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pickle

np.set_printoptions(suppress=True)

#-----------------------Model Class----------------------------

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv1d(5, 16, 11, stride=1, padding=5)
        self.conv2 = nn.Conv1d(16, 16, 11, stride=1, padding=5)
        self.drop1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv1d(16, 16, 11, stride=1, padding=5)
        self.conv4 = nn.Conv1d(16, 16, 11, stride=1, padding=5)
        self.drop2 = nn.Dropout(0.25)
        self.conv5 = nn.Conv1d(16, 16, 11, stride=1, padding=5)
        self.conv6 = nn.Conv1d(16, 16, 11, stride=1, padding=5)
        self.drop3 = nn.Dropout(0.25)
        
    
        x = torch.randn((5, 101)).view(-1, 5 , 101)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 500)
        self.fc2 = nn.Linear(500,2)
        
    def convs(self, x):
        
        x = F.max_pool1d(F.relu(self.conv2(F.relu(self.conv1(x)))), 2)
        x = (self.drop1(x))
        #print(x.shape)
        
        x = F.max_pool1d(F.relu(self.conv4(F.relu(self.conv3(x)))),  2)
        x = (self.drop2(x))
        #print(x.shape)
        
        x = F.max_pool1d(F.relu(self.conv6(F.relu(self.conv5(x)))),  2)
        x = (self.drop3(x))
        #print(x.shape)
        
        if self._to_linear is None:
           # print(x[0].shape)
            self._to_linear = x[0].shape[0]*x[0].shape[1]
            #print(self._to_linear)
        return x
        
                    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
#-----------------------Model Class----------------------------     

