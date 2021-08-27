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

#------initialize model and hyperparameters--------

net = Net()
print(net)


optimizer = optim.Adam(net.parameters(), lr=0.001)
#loss_function = nn.MSELoss()
loss_function = nn.BCELoss()

batch_size = 64
EPOCHS = 30

#------initialize model and hyperparameters--------
#--------initialize data for single fasta--------------

x = torch.Tensor([i[0] for i in data1]).view(-1, 5, 101)
y = torch.Tensor([i[1] for i in data1])

#--------initialize data for single fasta--------------
#----------------training for single fasta------------------

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(x), batch_size)):
        batchx = x[i : i+batch_size].view(-1, 5, 101)
        batchy = y[i : i+batch_size]


        net.zero_grad()
        
        outputs = net.forward(batchx)
        loss = loss_function(outputs, batchy)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch: {epoch}. Loss: {loss}")
    
#----------------training for single fasta------------------

#----------------------------Model Training--------------------------------

seqdataloader = data.DataLoader(concat_dataset, batch_size, shuffle = True)



for epoch in range(1, EPOCHS + 1):
    
    correct = 0
    all_label = []
    all_pred = []
    
    for i, (inputs, label) in enumerate(seqdataloader):
        
        
        inputs = inputs.view(-1, 5, 101)
        
        #print(inputs)
        #inputs = torch.Tensor(inputs[0])
        #print(len(inputs[0][0]))
        #inputs = inputs.view(-1, 5, 101)
        
        #print(label)
        net.zero_grad()
        outputs = net.forward(inputs)
        #print(outputs)
        loss = loss_function(outputs, label)
        loss.backward()
        optimizer.step()
        
        #---------------------AUC------------------------
        pred = outputs.argmax(dim=1, keepdim=True)
        real_labels = label.argmax(dim=1, keepdim= True)
        
        #for i in range(len(real_labels)):
            #print(real_labels[i], pred[i])
            
        correct += pred.eq(real_labels).sum().item()
        #print(correct)
        all_label.extend(real_labels.tolist())
        all_pred.extend(pred.tolist())
        #print(len(all_label))
        #------------------------------------------------
        
    #print(correct)
    #print(len(all_label))
    #print(correct/len(all_label))
    print(f"Epoch: {epoch}. Loss: {loss}")
    print("Train AUC score: {:.4f}".format(roc_auc_score(np.array(all_label), np.array(all_pred))))
    acc = correct/len(all_label)
    print('Accuracy: ' ,correct/len(all_label))
         
