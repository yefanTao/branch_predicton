# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:20:10 2019

@author: zs919
"""

import pdb
import os
from collections import deque
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.sparse.linalg
import datetime
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import utils

def sim(pred, file='gcc_branch.out', mod = 12, **kwargs, ):
    modtrace = {}
    branches = []
    with open(file, 'r') as file_in:
        for line in file_in:
            register = line[2:8]
            result = int(line[9])           
            intregister = int(register,16)
            modtraceInd = intregister%mod
            modtrace.setdefault(modtraceInd, []).append(result)
            branches.append([register, result])
    if file == 'gcc_branch.out':
        trainset = utils.GCC_Dataset_LSTM(file, train=True, mod=mod, l=kwargs['l'])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=False, num_workers=0)
        testset = utils.GCC_Dataset_LSTM(file, train=False, mod = mod,l=kwargs['l'])
        testloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=False, num_workers=0)
    if file == 'mcf_branch.out':
        trainset = utils.MCF_Dataset_LSTM(file, train=True, mod=mod,l=kwargs['l'])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=False, num_workers=0)
        testset = utils.MCF_Dataset_LSTM(file, train=False, mod = mod,l=kwargs['l'])
        testloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=False, num_workers=0)
    if pred == LSTM_predictor:
        LSTM_predictor(trainloader,testloader, mod = mod, l=kwargs['l'],w=kwargs['w'])
    else:
        num_correct = pred(branches, mod = mod, l=kwargs['l'])
        total = sum(len(r) for r in modtrace.values())
        return (num_correct * 1.0/total)


class Counter:
    state = 2   # 2-bit predictor, 1 and 2 predict do not take, 3 and 4 predict take
    def predict(self):
        if(self.state < 3):
            return -1
        if(self.state > 2):
            return 1

    def update(self, actual):
        if(actual == 1):
            self.state = self.state + 1
            if(self.state > 4):
                self.state = 4
        if(actual == -1):
            self.state = self.state - 1
            if(self.state < 1):
                self.state = 1
        return 


def local_predictor(trace, mod=9999999, l=1,w=0):# 2-bit

    c_list = {}
    num_correct = 0
    for br in trace:            # iterating through each branch
    #    print(type(br[0]))
        tmp = int(br[0],16)%mod
        if tmp not in c_list:     # if no previous branch from this memory location 
            c_list[tmp] = Counter()
        pr = c_list[tmp].predict()
        actual_value = 1 if br[1] else -1
        c_list[tmp].update(actual_value)
        if pr == actual_value:
            num_correct += 1
    return num_correct


# Perceptron
class Perceptron:
    weights = []
    N = 0
    bias = 0
    threshold = 0

    def __init__(self, N):
        self.N = N
        self.bias = 0
        self.threshold = 2 * N + 14                 # optimal threshold depends on history length
        self.weights = [0] * N      

    def predict(self, global_branch_history):
        running_sum = self.bias
        for i in range(0, self.N):                  # dot product of branch history with the weights
            running_sum += global_branch_history[i] * self.weights[i]
        prediction = -1 if running_sum < 0 else 1
        return (prediction, running_sum)

    def update(self, prediction, actual, global_branch_history, running_sum):
        if (prediction != actual) or (abs(running_sum) < self.threshold):   
            self.bias = self.bias + (1 * actual)
            for i in range(0, self.N):
                self.weights[i] = self.weights[i] + (actual * global_branch_history[i])

    def statistics(self):
        print("bias is: ",str(self.bias)," weights are: ",str(self.weights))

def perceptron_pred(trace,mod=9999999, l=1,w=0):

    global_branch_history = deque([])
    global_branch_history.extend([0]*l)

    p_list = {}
    num_correct = 0

    for br in trace:            # iterating through each branch
        tmp = int(br[0],16)%mod
        if tmp not in p_list:     # if no previous branch from this memory location 
            p_list[tmp] = Perceptron(l)
        results = p_list[tmp].predict(global_branch_history)
        pr = results[0]
        running_sum = results [1]
        actual_value = 1 if br[1] else -1
        p_list[tmp].update(pr, actual_value, global_branch_history, running_sum)
        global_branch_history.appendleft(actual_value)
        global_branch_history.pop()
        if pr == actual_value:
            num_correct += 1

    return num_correct

# Gshare
class Gshare:

    def __init__(self, l):
        self.state=[2]*(2**l)
    
    def predict(self, idx):
        if(self.state[idx] < 3):
            return -1
        if(self.state[idx] > 2):
            return 1

    def update(self, idx, actual):
        if(actual == 1):
            self.state[idx] = self.state[idx] + 1
            if(self.state[idx] > 4):
                self.state[idx] = 4
        if(actual == -1):
            self.state[idx] = self.state[idx] - 1
            if(self.state[idx] < 1):
                self.state[idx] = 1
        return 

def gshare_predictor(trace, mod=9999999, l=1,w=0):# 2-bit

    g = Gshare(l)
    c_history = [0]*l
    num_correct = 0
    for br in trace:            # iterating through each branch
        br_addr = int(br[0],16)%(2**l)
        br_hist = 0
        for i in range(len(c_history)):
            if c_history[i]:
                br_hist += 1<<(len(c_history)-i-1)
        gshare_addr=br_addr^br_hist
        pr = g.predict(gshare_addr)
        actual_value = 1 if br[1] else -1
        c_history.pop(0)
        c_history.append(actual_value) #update history
        g.update(gshare_addr,actual_value)  #update share predictor
        if pr == actual_value:
            num_correct += 1
    return num_correct

def LSTM_predictor(trace,testloader,mod=12,l=3,w=10):
    HIDDEN_DIM = 2    
    #model = utils.LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, mod, 2).float()   
    #loss_function = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    p_list = {}
    for epoch in range(20):
        start = datetime.datetime.now()
        num = 0
        correct = 0
        for sentence, tags in trace:
            #print(tags)
            i = tags[0][1].item()
            if i not in p_list:
                p_list[i] = [utils.LSTMTagger(l, 2).float(),nn.CrossEntropyLoss()]
                p_list[i].append(optim.SGD(p_list[i][0].parameters(), lr=0.1))
                p_list[i].append([])
                p_list[i].append([])
            if len(p_list[i][3])!=w:
                p_list[i][3].append(tags[0][0].item())
                p_list[i][4].append(sentence.numpy())
            else:
                label = torch.tensor(p_list[i][3])
                input = torch.tensor(p_list[i][4])
                input = torch.squeeze(input,1)
                p_list[i][0].zero_grad()
                p_list[i][0].hidden = p_list[i][0].init_hidden(2)
                tag_scores = p_list[i][0](input.float())
                pred = torch.argmax(tag_scores,1)
                for m in range(w):
                    if pred[m]==label[m]:
                        correct = correct+1
                loss = p_list[i][1](tag_scores, label)
                loss.backward()
                p_list[i][2].step()
                num = num+w
                
                p_list[i][3] = []
                p_list[i][4] = []
              
        end = datetime.datetime.now()
        #print(correct/num,end-start)
        Tnum = 0
        Tcorrect = 0
        for i in range(mod):
            p_list[i][0].hidden = p_list[i][0].init_hidden(2)
        for sentence,tags in testloader:
            i = tags[0][1].item()
            label = torch.tensor([tags[0][0].item()])
            input = torch.tensor(sentence.numpy())
            p_list[i][0].zero_grad()
            tag_scores = p_list[i][0](input.float())
            pred = torch.argmax(tag_scores,1)
            if pred==label:
                Tcorrect = Tcorrect+1
            Tnum = Tnum+1
        #print(Tcorrect/Tnum)
        print("                    %.5f             %.5f" % (correct/num, Tcorrect/Tnum)) 
    return 0



     


gcc = 'gcc_branch.out'
#gcc = 'test.out'
mcf = 'mcf_branch.out'
mod = 8#number of branches for local predictor
print('number of branches is',mod)
print("|Predictor|         |gcc accuracy|         |mcf accuracy|")
#nn_gcc = sim(local_predictor, file=gcc, mod=mod, l=2)
#nn_mcf = sim(local_predictor, file=mcf, mod=mod, l=1,w=10)
#print("local predictor        %.5f             %.5f" % (nn_gcc, nn_gcc))
#nn_gcc = sim(perceptron_pred, file=gcc,mod=mod, l=2)
#nn_mcf = sim(perceptron_pred, file=mcf,mod=mod, l=1,w=10)
print("perceptron (depth 8)   %.5f             %.5f" % (nn_gcc, nn_gcc))
nn_gcc = sim(gshare_predictor, file=gcc,mod=mod, l=3)
print("gshare                 %.5f             %.5f" % (nn_gcc, nn_gcc)) 
sim(LSTM_predictor, file=gcc,mod=mod, l=1,w=10)













    
