import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import os
import time
import copy
from tracin.tracin import save_tracin_checkpoint
from copy import deepcopy


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers=1, device="cpu"):
        super(LSTM, self).__init__()

        self.num_items = output_size
        self.device = device
        self.emb_length = input_size
        self.item_emb = nn.Embedding(self.num_items, self.emb_length, padding_idx=0)
        self.batch_size = 512

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

    def get_gradients(self):
        """Gets gradients of the model: To be used by tracin

        Returns:
            [type]: 1D torch tensor of gradients
        """
        list_params = list(self.parameters())
        # print("list_params \n",list_params)
        # print("grads \n", [l.grad for l in list_params])
        gradients = torch.cat(
            [torch.flatten(l.grad) for l in list_params if l.grad is not None]
        )
        return gradients

    def compute_metrics(self, test, test_labels):
        """Computes how well the model performs on the test set.

        Args:
            test ([type]): [description]
            test_labels ([type]): [description]

        Returns:
            [type]: [description]
        """
        test_num = len(test)
        MRR, HITS, loss = 0, 0, 0
        probs = {}  # variable used to store probabilities?
        criterion = nn.CrossEntropyLoss()
        for iteration in range(int(test_num / self.batch_size) + 1):
            st_idx, ed_idx = (
                iteration * self.batch_size,
                (iteration + 1) * self.batch_size,
            )
            if ed_idx > test_num:
                ed_idx = test_num
            output, hidden = self.forward(
                torch.stack([test[i][0] for i in range(st_idx, ed_idx)], dim=0).detach()
            )
            loss += criterion(output, test_labels[st_idx:ed_idx]).item()
            output1 = output.view(-1, self.num_items)
            prob = nn.functional.softmax(output1, dim=1).data.cpu()
            np_prob = prob.numpy()
            current_val = np.zeros((np_prob.shape[0], 1))
            for i in range(st_idx, ed_idx):
                current_test_label = test[i][1]
                current_val[i - st_idx, 0] = np_prob[i - st_idx, current_test_label]

            new_prob = np_prob - current_val
            ranks = np.count_nonzero(new_prob > 0, axis=1)

            for i in range(st_idx, ed_idx):
                predicted_rank = ranks[i - st_idx] + 1
                MRR += 1 / predicted_rank
                HITS += 1 if predicted_rank <= 10 else 0
                probs[i] = np_prob[i - st_idx, :]
        # print("Outputs shape is ", output.size())
        # print("outputs1 shape is ", output1.size())
        return MRR / test_num, HITS / test_num, loss / test_num, probs

    def traintest(self, train, test, epochs, learning_rate=5e-3, momentum=0.9):
        # NOTE: Added this deep copy
        train = deepcopy(train)
        test = deepcopy(test)

        train_num, test_num = len(train), len(test)
        train_labels, test_labels = [], []

        for i in range(train_num):
            train[i][0] = self.item_emb(torch.LongTensor(train[i][0]).to(self.device))
            train_labels.append(train[i][1])
        train_labels = torch.LongTensor(train_labels).to(self.device)

        for i in range(test_num):
            test[i][0] = self.item_emb(torch.LongTensor(test[i][0]).to(self.device))
            test_labels.append(test[i][1])
        test_labels = torch.LongTensor(test_labels).to(self.device)

        # print("train # = {}\ttest # = {}".format(train_num, test_num))
        print(f"train # = {train_num}, test # = {test_num}\n")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        start_time = time.time()


        train_losses = []
        test_losses = []
        test_hits = []
        test_mrr = []
        for epoch in range(0, epochs + 1):
            train_loss = 0
            for iteration in range(int(train_num / self.batch_size) + 1):
                st_idx, ed_idx = (
                    iteration * self.batch_size,
                    (iteration + 1) * self.batch_size,
                )
                if ed_idx > train_num:
                    ed_idx = train_num

                optimizer.zero_grad()
                output, hidden = self.forward(
                    torch.stack(
                        [train[i][0] for i in range(st_idx, ed_idx)], dim=0
                    ).detach()
                )
                loss = criterion(output, train_labels[st_idx:ed_idx])
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            if epoch % 5 == 0:
                test_MRR, test_HITS, test_loss, test_prediction = self.compute_metrics(
                    test, test_labels
                )
                print(
                    "Epoch {}\tTrain Loss: {}\tTest MRR: {}\tTest Recall@10: {}\tElapsed time: {}".format(
                        epoch,
                        train_loss / train_num,
                        test_MRR,
                        test_HITS,
                        time.time() - start_time,
                    )
                )
                # NOTE: append return values
                train_losses.append(train_loss / train_num)
                test_losses.append(test_loss)
                test_mrr.append(test_MRR)
                test_hits.append(test_HITS)

                start_time = time.time()
            if epoch % 10 == 0:
                path = os.getcwd()
                fname = path + "/checkpoints_v2/lstm_checkpoint_epoch" + str(epoch) + ".pt"
                print(f"saving checkpoint to {fname}")
                save_tracin_checkpoint(self, epoch, train_loss, optimizer, fname)


        return test_prediction, train_losses, test_losses, test_mrr, test_hits
