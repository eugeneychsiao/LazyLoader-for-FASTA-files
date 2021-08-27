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

class Data():
    def __init__(self):
        self.seq_length = 101
        self.data_path = "with_reverse/RBBP5"
        self.negative_list = []
        self.positive_list = []
        self.positivecount = 0
        self.negativecount = 0
        self.onehot_encoded_positive = []
        self.onehot_encoded_negative = []
        
    def onehot_encode_sequences(self, sequences):
        onehot = []
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N':4}
        for sequence in sequences:
            arr = np.zeros((len(sequence), 5)).astype("float")
            for (i, letter) in enumerate(sequence):
                arr[i, mapping[letter]] = 1.0
            onehot.append(arr)
        return onehot

    def reverse_complement(self, dna):
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A','N': 'N'}
        return ''.join([complement[base] for base in dna[::-1]])
        
        
        
    def fasta_prepare_data(self):
        path = os.path.join(self.data_path, "RBBP5_positive_training.fa")
        f = open(path, 'r')
        for line in f:
            self.positive_list.append(line)
            self.positivecount += 1
        #print(self.positivecount)
        self.positive_list = list(map(lambda x:x.strip(), self.positive_list))
        onehot_positive = self.onehot_encode_sequences(self.positive_list)


        path = os.path.join(self.data_path, "RBBP5_negative_training.fa")
        f = open(path, 'r')
        for line in f:
            self.negative_list.append(line)
            self.negativecount += 1
        #print(self.negativecount)
        self.negative_list = list(map(lambda x:x.strip(), self.negative_list))
        onehot_negative = self.onehot_encode_sequences(self.negative_list)
        
        combinedlist = []
        
        for sequence in onehot_positive:
            combinedlist.append([sequence, [0,1]])
        for sequence in onehot_negative:
            combinedlist.append([sequence,[1,0]])
        
        np.random.shuffle(combinedlist)
        
        
        for data in combinedlist:
            data[0] = np.transpose(data[0])
            
        
            
        return combinedlist
    
    

    def seq_prepare_data(self):
        #----------------------------------------------------------------------------------------------
        # positive data
        path = "/usr/local/lib/EzGeno/Backbone_model_data/"
        file_counter = 0
        
        for file in os.listdir(path):
            new_path = os.path.join(path, file)


            if new_path.lower().endswith('seq'):
                file_counter += 1
                #print("Number of files processed: %s" %(file_counter))
                # print(new_path)
                with open(new_path) as outputfile:
                    for sequence in outputfile:

                        if sequence[0] != 'A':
                            pass
                        else:
                            self.positive_list.append(sequence[17:118])
                            self.positive_list.append(self.reverse_complement(sequence[17:118]))

        print("shuffling")                    
        random.shuffle(self.positive_list)
        print("Number of positive sequences: %s" %(len(self.positive_list)))


        #-----------------------------------------------------------------------------------------------
        #negative data

        negative_path = "/usr/local/lib/EzGeno/negative_fasta.fa"
        test_sequences = SeqIO.parse(open('negative_fasta.fa'),'fasta')
        with open('negative_fasta.fa') as out_file:
            for sequence in test_sequences:
                self.negative_list.append(str(sequence.seq).upper())

        print("Number of negative sequences: %s" %(len(self.negative_list)))
        
        
        return self.positive_list, self.negative_list

        
        
my_data = Data()
data1 = my_data.fasta_prepare_data()
#positive_list, negative_list = my_data.seq_prepare_data()




