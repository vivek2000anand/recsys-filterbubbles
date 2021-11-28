import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from os import listdir
from os.path import isfile, join
from tracin.tracin import (
    save_tracin_checkpoint,
    load_tracin_checkpoint,
    calculate_tracin_influence,
)
import pandas as pd
from LSTM_clean.utils import train_test_split, sequence_generator
from LSTM_clean.model import LSTM


def printl():
    print(80 * "-")


curr_dir = os.getcwd()
path = curr_dir + "/checkpoints/"
checkpoints = []
with os.scandir(path) as listOfEntries:
    for entry in listOfEntries:
        # print all entries that are files
        if entry.is_file():
            checkpoints.append(os.path.join(path, entry.name))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cpu")
print("Device is ", device)


print("Loading data")
path = os.getcwd()
raw_data = pd.read_csv(path + "/data/twitch100k.csv", sep=",", header=0)
print("Data looks like \n", raw_data.head())
streamer_community_dict = pd.Series(
    raw_data["community"].values, index=raw_data["streamer_name"]
).to_dict()
data = raw_data[["user_id", "streamer_name", "stop_time"]]
print("Cleaned data looks like \n", data.head())
data = data.values
look_back = 50

unique_users = sorted(list(set(data[:, 0])))
unique_items = sorted(list(set(data[:, 1])))

user_dic = {user: idx for (idx, user) in enumerate(unique_users)}
item_dic = {item: idx for (idx, item) in enumerate(unique_items)}

for (idx, row) in enumerate(data):
    user, item, time = user_dic[row[0]], item_dic[row[1]], row[2]
    data[idx, 0], data[idx, 1] = int(user), int(item)

# print(data)
print("Preprocessing Data")
original_data = train_test_split(data=data)
train, test, train_items, test_items = sequence_generator(original_data, look_back)
test_ground_truth = {i: test[i][1] for i in range(len(test))}

print("Train: {}, Test: {}".format(len(train), len(test)))

train_num, test_num = len(train), len(test)
train_labels, test_labels = [], []


# for i in range(train_num):
#     train[i][0] = model.item_emb(torch.LongTensor(train[i][0]).to(model.device))
#     train_labels.append(train[i][1])
# train_labels = torch.LongTensor(train_labels).to(model.device)

# for i in range(test_num):
#     test[i][0] = model.item_emb(torch.LongTensor(test[i][0]).to(model.device))
#     test_labels.append(test[i][1])
# test_labels = torch.LongTensor(test_labels).to(model.device)

print("__________________________________________________________________________")


source = torch.LongTensor(train[3][0]).to(device)
source_label = torch.LongTensor([train[3][1]]).to(device)

target = torch.LongTensor(test[1][0]).to(device)
target_label = torch.LongTensor([test[1][1]]).to(device)

print("Source is ", source)
print("Source label is ", source_label)
print("Target is ", target)
print("Target Label is ", target_label)

print("Calculating Influence \n __________________________________")
influence = calculate_tracin_influence(
    LSTM, source, source_label, target, target_label, "SGD", checkpoints
)
print("Influence is ", influence)

print("__________________________________________________________________________")

source = torch.LongTensor(train[100][0]).to(device)
source_label = torch.LongTensor([train[100][1]]).to(device)

target = torch.LongTensor(test[5][0]).to(device)
target_label = torch.LongTensor([test[5][1]]).to(device)

print("Source is ", source)
print("Source label is ", source_label)
print("Target is ", target)
print("Target Label is ", target_label)

print("Calculating Influence \n __________________________________")
influence = calculate_tracin_influence(
    LSTM, source, source_label, target, target_label, "SGD", checkpoints
)
print("Influence is ", influence)

print("__________________________________________________________________________")


source = torch.LongTensor(train[2][0]).to(device)
source_label = torch.LongTensor([train[2][1]]).to(device)

target = torch.LongTensor(test[6][0]).to(device)
target_label = torch.LongTensor([test[6][1]]).to(device)

print("Source is ", source)
print("Source label is ", source_label)
print("Target is ", target)
print("Target Label is ", target_label)

print("Calculating Influence \n __________________________________")
influence = calculate_tracin_influence(
    LSTM, source, source_label, target, target_label, "SGD", checkpoints
)
print("Influence is ", influence)

print("__________________________________________________________________________")


source = torch.LongTensor(train[209][0]).to(device)
source_label = torch.LongTensor([train[209][1]]).to(device)

target = torch.LongTensor(test[4][0]).to(device)
target_label = torch.LongTensor([test[4][1]]).to(device)

print("Source is ", source)
print("Source label is ", source_label)
print("Target is ", target)
print("Target Label is ", target_label)

print("Calculating Influence \n __________________________________")
influence = calculate_tracin_influence(
    LSTM, source, source_label, target, target_label, "SGD", checkpoints
)
print("Influence is ", influence)


print("\n\nSelf-influence!")
source = torch.LongTensor(train[209][0]).to(device)
source_label = torch.LongTensor([train[209][1]]).to(device)

source = torch.LongTensor(test[4][0]).to(device)
source_label = torch.LongTensor([test[4][1]]).to(device)

print("Source is ", source)
print("Source label is ", source_label)
print("Target is ", target)
print("Target Label is ", target_label)

print("Calculating Influence \n __________________________________")
influence = calculate_tracin_influence(
    LSTM, source, source_label, target, target_label, "SGD", checkpoints
)
print("Influence is ", influence)

# %%
