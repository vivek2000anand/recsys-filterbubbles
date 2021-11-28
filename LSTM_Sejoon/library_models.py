"""
This is a supporting library with the code of the model.
"""

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
from collections import defaultdict
import os
import time
import random


class LSTM(nn.Module):
    def __init__(
        self, data, input_size, output_size, hidden_dim, n_layers=1, device="cpu"
    ):
        super(LSTM, self).__init__()

        self.num_items = output_size
        self.device = device
        self.emb_length = input_size
        self.item_emb = nn.Embedding(self.num_items, self.emb_length, padding_idx=0)
        self.batch_size = 1024

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
        hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim)
            .to(self.device)
            .detach(),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim)
            .to(self.device)
            .detach(),
        )
        return hidden

    def compute_metrics(self, test, test_out):
        total_test_num = len(test)
        criterion = nn.CrossEntropyLoss()

        MRR, HITS, test_loss = 0, 0, 0
        for iteration in range(int(total_test_num / self.batch_size) + 1):
            st_idx, ed_idx = (
                iteration * self.batch_size,
                (iteration + 1) * self.batch_size,
            )
            if ed_idx > total_test_num:
                ed_idx = total_test_num
            output, hidden = self.forward(
                torch.stack([test[i][0] for i in range(st_idx, ed_idx)], dim=0).detach()
            )
            loss = criterion(output, test_out[st_idx:ed_idx])
            test_loss += loss.item()
            loss.backward()  # Does backpropagation and calculates gradients

            output = output.view(-1, self.num_items)
            prob = nn.functional.softmax(output, dim=1).data.cpu()
            np_prob = prob.numpy()
            current_val = np.zeros((np_prob.shape[0], 1))
            for i in range(st_idx, ed_idx):
                current_test_label = test[i][1]
                current_val[i - st_idx, 0] = np_prob[i - st_idx, current_test_label]

            new_prob = np_prob - current_val
            ranks = np.count_nonzero(new_prob > 0, axis=1)

            for i in range(st_idx, ed_idx):
                rank = ranks[i - st_idx] + 1
                MRR += 1 / rank
                HITS += 1 if rank <= 10 else 0

        MRR /= total_test_num
        HITS /= total_test_num
        return MRR, HITS, test_loss

    def traintest(self, train, valid, test, epochs):

        start_time = time.time()
        total_train_num = len(train)
        current_labels = []
        for i in range(total_train_num):
            train[i][0] = self.item_emb(torch.LongTensor(train[i][0]).to(self.device))
            current_labels.append(train[i][1])
        train_out = torch.LongTensor(current_labels).to(self.device)

        print("train #={}".format(total_train_num))

        total_valid_num = len(valid)
        current_labels = []
        for i in range(total_valid_num):
            valid[i][0] = self.item_emb(torch.LongTensor(valid[i][0]).to(self.device))
            current_labels.append(valid[i][1])
        valid_out = torch.LongTensor(current_labels).to(self.device)

        print("valid #={}".format(total_valid_num))

        total_test_num = len(test)
        current_labels = []
        for i in range(total_test_num):
            test[i][0] = self.item_emb(torch.LongTensor(test[i][0]).to(self.device))
            current_labels.append(test[i][1])
        test_out = torch.LongTensor(current_labels).to(self.device)

        print("test #={}".format(total_test_num))

        criterion = nn.CrossEntropyLoss()
        learning_rate = 1e-3
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        MRR, HITS = 0, 0

        patience = 10
        best_loss, best_MRR, best_HITS, best_epoch = 2147483647, 0, 0, 0
        for epoch in range(epochs):

            train_loss = 0
            for iteration in range(int(total_train_num / self.batch_size) + 1):
                st_idx, ed_idx = (
                    iteration * self.batch_size,
                    (iteration + 1) * self.batch_size,
                )
                if ed_idx > total_train_num:
                    ed_idx = total_train_num

                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                output, hidden = self.forward(
                    torch.stack(
                        [train[i][0] for i in range(st_idx, ed_idx)], dim=0
                    ).detach()
                )
                loss = criterion(output, train_out[st_idx:ed_idx])
                loss.backward()  # Does backpropagation and calculates gradients
                train_loss += loss.item()
                optimizer.step()  # Updates the weights accordingly

            if epoch % 5 == 0:
                val_MRR, val_HITS, val_loss = self.compute_metrics(valid, valid_out)
                test_MRR, test_HITS, test_loss = self.compute_metrics(test, test_out)

                print(
                    "Epoch {}\tTrain Loss: {}\tVal Loss = {}\tTest MRR = {}\tTest Recall@10 = {}\tElapsed time: {}".format(
                        epoch,
                        train_loss / total_train_num,
                        val_loss,
                        test_MRR,
                        test_HITS,
                        time.time() - start_time,
                    )
                )
                if best_loss > val_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_MRR, best_HITS = test_MRR, test_HITS
                elif epoch - best_epoch >= patience:
                    break

                start_time = time.time()

        print(
            "Test MRR = {}\tTest HITS = {} at Epoch {}\n".format(
                best_MRR, best_HITS, best_epoch
            )
        )
        return best_MRR, best_HITS
