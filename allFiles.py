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

class allfiles(data.Dataset):
    
    
    def __init__(self, pos_data_dir, neg_data_dir):
        
        self.list_index_counter = 0
        self.combined_list = []
        self.pos_filename = os.listdir(pos_data_dir)
        self.neg_filename = os.listdir(neg_data_dir)
        self.num_files = len(self.pos_filename)
        self.pos_filepath = os.path.join(pos_data_dir, self.pos_filename[self.list_index_counter]) 
        self.neg_filepath = os.path.join(neg_data_dir, self.neg_filename[self.list_index_counter])
        
        f = open(self.pos_filepath, 'r')
        for line in f:
            temp_seq = self.one_hot(line.strip('\n'))
            self.combined_list.append([np.transpose(torch.Tensor(temp_seq)), torch.Tensor([0,1])])
            
        f = open(self.neg_filepath, 'r') 
        for line in f:
            temp_seq = self.one_hot(line.strip('\n'))
            self.combined_list.append([np.transpose(torch.Tensor(temp_seq)), torch.Tensor([1,0])])
        
        self.list_index_counter += 1    
        random.shuffle(self.combined_list)        
        
    def one_hot(self, seq):
        encoding_map = {'A': [1,0,0,0,0], 'T': [0,1,0,0,0], 'C': [0,0,1,0,0], 'G': [0,0,0,1,0], 'N': [0,0,0,0,1]}
        temp_list = []
        for s in seq:
            temp_list.append(encoding_map[s])
        return temp_list
               
    def __getitem__(self, index):
        if self.list_index_counter < self.num_files:
            if len(self.combined_list) < 15000:
                self.pos_filepath = os.path.join(pos_data_dir, self.pos_filename[self.list_index_counter]) 
                self.neg_filepath = os.path.join(neg_data_dir, self.neg_filename[self.list_index_counter])
                f = open(self.pos_filepath, 'r')
                for line in f:
                    temp_seq = self.one_hot(line.strip('\n'))
                    self.combined_list.append([np.transpose(torch.Tensor(temp_seq)), torch.Tensor([0,1])])

                f = open(self.neg_filepath, 'r') 
                for line in f:
                    temp_seq = self.one_hot(line.strip('\n'))
                    self.combined_list.append([np.transpose(torch.Tensor(temp_seq)), torch.Tensor([1,0])])
                self.list_index_counter += 1
            
            
        random.shuffle(self.combined_list) 
        selected_index = self.combined_list[index]
        self.combined_list.pop(index)
         
        return selected_index

    def __len__(self):
        return len(self.combined_list)
    
test = allfiles('processed/positive', 'processed/negative')
print(len(test.combined_list))
test.__getitem__(5)
print(len(test.combined_list))
