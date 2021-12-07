import itertools
import numpy as np
import pickle
from itertools import product
import os
import torch
from torch.cuda import random
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from os import listdir
from os.path import isfile, join
from tracin.tracin_batched import save_tracin_checkpoint, load_tracin_checkpoint,  approximate_tracin_batched
import pandas as pd
from LSTM_clean.utils import train_test_split, sequence_generator, get_diversity
from LSTM_clean.model import LSTM
import re
from statistics import mean
import scipy.stats as stats
from sklearn.utils import shuffle
from copy import deepcopy
import time

OUTPUT_SIZE = 3312

def get_checkpoints():
    curr_dir = os.getcwd()
    path = curr_dir + "/checkpoint_subsets/"
    checkpoints = []
    with os.scandir(path) as listOfEntries:
        for entry in listOfEntries:
            # print all entries that are files
            if entry.is_file():
                checkpoints.append(os.path.join(path,entry.name))
    return checkpoints


def get_train_validation():
    cwd = os.getcwd()
    test_path = cwd + "/data/test.data"
    train_path = cwd + "/data/train.data"
    valid_path = cwd + "/data/valid.data"

    train_data = np.load(train_path, allow_pickle=True)
    valid_data = np.load(valid_path, allow_pickle=True)
    train = [t[0] for t in train_data]
    train_labels = [t[1] for t in train_data]
    valid = [t[0] for t in valid_data]
    valid_labels = [t[1] for t in valid_data]
    return train, train_labels, valid, valid_labels

def get_valid_subset(length, valid, valid_labels, valid_lengths):
    valid_subset =[]
    valid_labels_subset = []
    for i in range(len(valid)):
        if valid_lengths[i] == length:
            valid_subset.append(valid[i])
            valid_labels_subset.append(valid_labels[i])
    return valid_subset, valid_labels_subset

def get_length(data_point):
    for i in range(len(data_point)):
        if data_point[i] != 0:
            return 49 -i
    return 0


def get_points(x, x_label, y, y_labels):
    combos = list(product(zip(valid_subset, valid_labels_subset), zip(train, train_labels)))
    sources = [c[0][0] for c in combos]
    source_labels = [c[0][1] for c in combos]
    targets = [c[1][0] for c in combos]
    target_labels = [c[1][1] for c in combos]
    return sources, source_labels, targets, target_labels

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='7'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is ", device)

checkpoints = get_checkpoints()
train, train_labels, valid, valid_labels = get_train_validation()
valid_lengths = [get_length(i) for i in valid]

influences = []
start_time = time.time()
for i in range(50):
    start_length_time = time.time()
    valid_subset, valid_labels_subset = get_valid_subset(i, valid, valid_labels, valid_lengths)
    if len(valid_subset) != 0:
        sources, source_labels, targets, target_labels = get_points(valid, valid_labels, train, train_labels)
        influence = approximate_tracin_batched(LSTM, sources=sources, targets=targets, source_labels=source_labels, target_labels=train_labels, optimizer="SGD", paths=checkpoints, batch_size=4000, num_items=OUTPUT_SIZE, device=device)
        influences.append(influence)
        end_length_time = time.time()
        print(f"Influence for length {i} is : {influence} \nTime elapsed {end_length_time-start_length_time}")
    else:
        influences.append(-1)

print(f"Influences are \n{influences}")