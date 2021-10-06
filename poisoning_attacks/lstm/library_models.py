'''
This is a supporting library with the code of the model.
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
from itertools import chain
import time
import random
from scipy.stats import entropy
import rbo
from scipy import stats
from library_data import *
from tqdm import trange
import library_models as lib
import copy

total_reinitialization_count=0

def mean_confidence_interval(data, confidence=0.95):
        
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


class LSTM(nn.Module):
    def __init__(self, data, input_size, output_size, hidden_dim, n_layers=1, device="cpu",seed=0):
        super(LSTM, self).__init__()
        
        self.num_items = output_size
        self.device = device 
        self.emb_length = input_size
        self.item_emb = nn.Embedding(self.num_items, self.emb_length,padding_idx=0)
        self.batch_size = 1024
        self.random_seed = seed

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # LSTM Layer
        self.LSTM = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.LSTM(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        inp = out[:, -1, :].contiguous().view(-1, self.hidden_dim)
        out = self.fc(inp)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device).detach(), torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device).detach())
        return hidden

    def traintest(self, train,test,epochs, original_probs,original_rank,final_metrics):

        
        total_train_num = len(train)
        current_labels = []
        for i in range(total_train_num):
            train[i][0] = self.item_emb(torch.LongTensor(train[i][0]).to(self.device))
            current_labels.append(train[i][1])
        train_out = torch.LongTensor(current_labels).to(self.device)
        
        print("train #={}".format(total_train_num))

        criterion = nn.CrossEntropyLoss()
        learning_rate = 1e-3
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        prev_loss = 2147483647
        start_time = time.time()

        total_test_num = len(test)
        current_labels = []
        for i in range(total_test_num):
            test[i][0] = self.item_emb(torch.LongTensor(test[i][0]).to(self.device))
            current_labels.append(test[i][1])
        test_out = torch.LongTensor(current_labels).to(self.device)

        print("test #={}".format(total_test_num))

        # METRICS = [MRR_diff, HITS_diff, RBO_diff, KL_div_diff]
        predicted_rank = [0 for i in range(total_test_num)]
        probs = [0 for i in range(total_test_num)]

        MRR,HITS = 0,0

        for epoch in range(epochs):
            
            for iteration in range(int(total_test_num / self.batch_size) + 1):
                st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
                if ed_idx > total_test_num:
                    ed_idx = total_test_num
                output, hidden = self.forward(torch.stack([test[i][0] for i in range(st_idx,ed_idx)],dim=0).detach())
                test_loss = criterion(output, test_out[st_idx:ed_idx])
                test_loss.backward()  # Does backpropagation and calculates gradients

                if epoch==epochs-1:
                    output = output.view(-1, self.num_items)
                    prob = nn.functional.softmax(output, dim=1).data.cpu()
                    np_prob = prob.numpy()
                    current_val = np.zeros((np_prob.shape[0],1))
                    for i in range(st_idx,ed_idx):
                        current_test_label = test[i][1]
                        current_val[i-st_idx,0] = np_prob[i-st_idx,current_test_label]
                    
                    new_prob = np_prob - current_val
                    ranks = np.count_nonzero(new_prob>0,axis=1)
                    
                    for i in range(st_idx,ed_idx):
                        predicted_rank[i] = ranks[i-st_idx]+1 
                        MRR += 1/predicted_rank[i]
                        HITS += (1 if predicted_rank[i]<=10 else 0)
                        probs[i] = np_prob[i-st_idx,:]

            if original_probs!=-1 and epoch==epochs-1:
                rank1,rank2 = np.argsort(original_probs,axis=1)[:,::-1],np.argsort(probs,axis=1)[:,::-1]
                for i in range(total_test_num):
                    if test[i][2]==1:
                        ground_truth = test[i][1]
                        MRR_diff = abs((1/predicted_rank[i])-(1/original_rank[i]))
                        HITS_diff = abs((1 if predicted_rank[i]<=10 else 0)-(1 if original_rank[i]<=10 else 0))
                        RBO = rbo.RankingSimilarity(rank1[i,:], rank2[i,:]).rbo()
                        rank_diff = abs(predicted_rank[i]-original_rank[i])
                        prob_diff = abs(probs[i][ground_truth] - original_probs[i][ground_truth])
                        jaccard = np.intersect1d(rank1[i,:10],rank2[i,:10]).shape[0]/np.union1d(rank1[i,:10],rank2[i,:10]).shape[0]
                        final_metrics[0].append(MRR_diff)
                        final_metrics[1].append(HITS_diff)
                        final_metrics[2].append(RBO)
                        final_metrics[3].append(rank_diff)
                        final_metrics[4].append(prob_diff)
                        final_metrics[5].append(jaccard)
           
            train_loss=0
            for iteration in range(int(total_train_num/self.batch_size)+1):
                st_idx,ed_idx = iteration*self.batch_size, (iteration+1)*self.batch_size
                if ed_idx>total_train_num:
                    ed_idx = total_train_num
                        
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                output, hidden = self.forward(torch.stack([train[i][0] for i in range(st_idx,ed_idx)],dim=0).detach())
                loss = criterion(output, train_out[st_idx:ed_idx])
                loss.backward()  # Does backpropagation and calculates gradients
                train_loss += loss.item()
                optimizer.step()  # Updates the weights accordingly
                 
            if epoch % 10 == 0:
                print("Epoch {}\tTrain Loss: {}\tElapsed time: {}".format(epoch, train_loss/total_train_num, time.time() - start_time))
                start_time = time.time()
        
        MRR /= total_test_num
        HITS /= total_test_num
        print('Unperturbed/Total Test num = {}/{}\tMRR = {}\tHITS = {}\n'.format(len(final_metrics[0]),total_test_num,MRR,HITS))
        return [probs,predicted_rank,final_metrics,[MRR,HITS]]

