#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:37:39 2019

@author: ziyushu
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class GCC_Dataset(torch.utils.data.Dataset):
    def __init__(self,file = 'gcc_branch.out',train = True,mod=12,l=10):
        BranchInd = []
        Result = []        
        with open(file, 'r') as file_in:
            for line in file_in:
                register = line[2:8]
                result = int(line[9])           
                intregister = int(register,16)
                modtraceInd = intregister%mod
                BranchInd.append(modtraceInd-mod/2)
                Result.append(result)
        length = len(BranchInd)
        tmp = BranchInd[-l+1:]
        tmp = tmp + BranchInd
        tmpresult = Result[-l+1:]
        tmpresult = tmpresult + Result
        if train:
            x = np.zeros((length//2,2*l-1))
            y = np.zeros(length//2) 
            for i in range(length//2):
                x[i,0:l] = tmp[i:i+l]
                x[i,l:] = tmpresult[i:i+l-1]
                y[i] = Result[i]
        else:
            x = np.zeros((length - length//2,2*l-1))
            y = np.zeros(length - length//2)
            for i in range(length//2,length):
                x[i-length//2,0:l] = tmp[i:i+l]
                x[i-length//2,l:] = tmpresult[i:i+l-1]
                y[i-length//2] = Result[i]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y).long()
        self.len = self.x_data.shape[0]  
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    

    
class MCF_Dataset(torch.utils.data.Dataset):
    def __init__(self,file = 'mcf_branch.out',train = True,mod=12,l=10):
        BranchInd = []
        Result = []        
        with open(file, 'r') as file_in:
            for line in file_in:
                register = line[2:8]
                result = int(line[9])           
                intregister = int(register,16)
                modtraceInd = intregister%mod
                BranchInd.append(modtraceInd-l/2)
                Result.append(result)
        length = len(BranchInd)
        tmp = BranchInd[-l+1:]
        tmp = tmp + BranchInd
        tmpresult = Result[-l+1:]
        tmpresult = tmpresult + Result
        if train:
            x = np.zeros((length//2,2*l-1))
            y = np.zeros(length//2) 
            for i in range(length//2):
                x[i,0:l] = tmp[i:i+l]
                x[i,l:] = tmpresult[i:i+l-1]
                y[i] = Result[i]
        else:
            x = np.zeros((length - length//2,2*l-1))
            y = np.zeros(length - length//2)
            for i in range(length//2,length):
                x[i-length//2,0:l] = tmp[i:i+l]
                x[i-length//2,l:] = tmpresult[i:i+l-1]
                y[i-length//2] = Result[i]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y).long()
        self.len = self.x_data.shape[0]  
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class GCC_Dataset_LSTM(torch.utils.data.Dataset):
    def __init__(self,file = 'gcc_branch.out',train = True,mod=12,l=3):
        BranchInd = []
        Result = []        
        with open(file, 'r') as file_in:
            for line in file_in:
                register = line[2:8]
                result = int(line[9])           
                intregister = int(register,16)
                modtraceInd = intregister%mod
                BranchInd.append(modtraceInd)
                Result.append(result)
        length = len(BranchInd)
        circ_result = Result[-l:]
        circ_result = circ_result + Result
        if train:
            x = np.zeros((length//2,1,l))
            y = np.zeros((length//2,2)) 
            for i in range(length//2):
                x[i,0,:] = circ_result[i:i+l]
                y[i,0] = Result[i]
                y[i,1] = BranchInd[i]
        else:
            x = np.zeros((length - length//2,1,l))
            y = np.zeros((length - length//2,2))
            for i in range(length//2,length):
                x[i-length//2,0,:] = circ_result[i-length//2:i-length//2+l]
                y[i-length//2,0] = Result[i-length//2]
                y[i-length//2,1] = BranchInd[i-length//2]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y).long()
        self.len = self.x_data.shape[0]  
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class MCF_Dataset_LSTM(torch.utils.data.Dataset):
    def __init__(self,file = 'mcf_branch.out',train = True,mod=12,l=3):
        BranchInd = []
        Result = []        
        with open(file, 'r') as file_in:
            for line in file_in:
                register = line[2:8]
                result = int(line[9])           
                intregister = int(register,16)
                modtraceInd = intregister%mod
                BranchInd.append(modtraceInd)
                Result.append(result)
        length = len(BranchInd)
        circ_result = Result[-l:]
        circ_result = circ_result + Result
        if train:
            x = np.zeros((length//2,1,l))
            y = np.zeros((length//2,2)) 
            for i in range(length//2):
                x[i,0,:] = circ_result[i:i+l]
                y[i,0] = Result[i]
                y[i,1] = BranchInd[i]
        else:
            x = np.zeros((length - length//2,1,l))
            y = np.zeros((length - length//2,2))
            for i in range(length//2,length):
                x[i-length//2,0,:] = circ_result[i-length//2:i-length//2+l]
                y[i-length//2,0] = Result[i-length//2]
                y[i-length//2,1] = BranchInd[i-length//2]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y).long()
        self.len = self.x_data.shape[0]  
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        #self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    def init_hidden(self, hidden_dim,batch):
        return(torch.zeros(1, batch, hidden_dim),torch.zeros(1, batch, hidden_dim))
    def forward(self, sentence):
        #embeds = self.word_embeddings(sentence)
        #lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        #tmp = sentence.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(sentence,self.hidden)
        #tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.softmax(lstm_out,dim=2)
        return tag_scores,lstm_out
