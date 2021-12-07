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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from copy import deepcopy
import time

OUTPUT_SIZE = 3312
NUM_TRAIN_SAMPLES = 20
NUM_VAL_SAMPLES = 20
NUM_REPETITIONS = 20
STEP_SIZE = 2
BATCH_SIZE = 4096
TRAIN_NAME = "filter"
TEST_NAME = "breaking"

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


def get_train_validation(train_dataset, valid_dataset):
    cwd = os.getcwd()
    test_path = cwd + "/data/test.data"
    if train_dataset == "filter":
        train_path = cwd + "/data/train_pts_filter_bubble.data"
    elif train_dataset == "diverse":
        train_path = cwd + "/data/train_pts_diverse.data"
    else:
        train_path = cwd + "/data/train_pts_breaking_bubble.data"
    # Valid path
    if valid_dataset == "filter":
        valid_path = cwd + "/data/filter_bubble_pred_pts.data"
    else:
        valid_path = cwd + "/data/breaking_bubble_pred_pts.data"

    train_data = np.load(train_path, allow_pickle=True)
    valid_data = np.load(valid_path, allow_pickle=True)
    train = [t[0] for t in train_data]
    train_labels = [t[1] for t in train_data]
    valid = [t[0] for t in valid_data]
    valid_labels = [t[1] for t in valid_data]
    print(f"Train set is length: {len(train)}")
    print(f"Validation set is length: {len(valid)}")
    return train, train_labels, valid, valid_labels

def get_train_subset(length, x, x_labels, train_lengths):
    x_subset =[]
    x_labels_subset = []
    for i in range(len(x)):
        if train_lengths[i] == length:
            x_subset.append(x[i])
            x_labels_subset.append(x_labels[i])
    # x_subset, _, x_labels_subset, _ = train_test_split(x_subset, x_labels_subset, train_size=num_sample, random_state=seed)
    # x_subset, x_labels_subset = shuffle(x_subset, x_labels_subset, random_state=seed)
    # if len(x_subset) > subset_size:
    #     return x_subset[:subset_size], x_labels_subset[:subset_size]
    # else:
    return x_subset, x_labels_subset

def get_length(data_point):
    for i in range(len(data_point)):
        if data_point[i] != 0:
            return 49 -i
    return 0


def get_points(x, x_label, y, y_label,x_num_sample =100, y_num_sample=10, seed=69):
    x, x_label = shuffle(x, x_label, random_state= seed)
    y, y_label = shuffle(y, y_label, random_state=seed)
    x, x_label = x[:x_num_sample], x_label[:x_num_sample]
    y, y_label = y[:y_num_sample], y_label[:y_num_sample]
    combos = list(product(zip(x, x_label), zip(y, y_label)))
    sources = [c[0][0] for c in combos]
    source_labels = [c[0][1] for c in combos]
    targets = [c[1][0] for c in combos]
    target_labels = [c[1][1] for c in combos]
    print(f"Number of datapoints {len(sources)}")
    return sources, source_labels, targets, target_labels

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is ", device)

checkpoints = get_checkpoints()
train, train_labels, valid, valid_labels = get_train_validation(train_dataset=TRAIN_NAME, valid_dataset=TEST_NAME)
train_lengths = [get_length(i) for i in train]

influences = {i:[] for i in range(0,50,STEP_SIZE)}
start_time = time.time()
print("About to start running")
for h in range(NUM_REPETITIONS):
    outer_start_time = time.time()
    print(f"Starting outer loop with {h}")
    for i in range(0, 50, STEP_SIZE):
        start_length_time = time.time()
        train_subset, train_labels_subset = get_train_subset(i, train, train_labels, train_lengths)
        if len(train_subset) != 0:
            print("About to cartesian product")
            sources, source_labels, targets, target_labels = get_points(train_subset, train_labels_subset, valid, valid_labels, x_num_sample=NUM_TRAIN_SAMPLES, y_num_sample=NUM_VAL_SAMPLES, seed=h)
            print("About to tracin")
            influence = approximate_tracin_batched(LSTM, sources=sources, targets=targets, source_labels=source_labels, target_labels=train_labels, optimizer="SGD", paths=checkpoints, batch_size=BATCH_SIZE, num_items=OUTPUT_SIZE, device=device)
            influences[i].append(influence)
            end_length_time = time.time()
            print(f"Influence for length {i} is : {influence} \nTime elapsed {end_length_time-start_length_time}")
        else:
            influences[i].append(-1)
    outer_end_time = time.time()
    print(f"Outer Iteration {h} has ended with {outer_end_time-outer_start_time} time taken")
    print("_______________________________________________________________________________")

print(f"Influences are \n{influences}")


for key, val in influences.items():
    influences[key] =[float(v) for v in val]


file_name = "train_"+ TRAIN_NAME + "_test_" + TEST_NAME +".pkl" 

with open(file_name, 'wb') as f:
    pickle.dump(influences, f)