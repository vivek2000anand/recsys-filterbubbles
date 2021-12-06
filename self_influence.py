import torch
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
import numpy as np
import re
from statistics import mean


curr_dir = os.getcwd()
path = curr_dir + "/checkpoints/"
checkpoints = []
with os.scandir(path) as listOfEntries:
    for entry in listOfEntries:
        # print all entries that are files
        if entry.is_file():
            checkpoints.append(os.path.join(path,entry.name))

last_checkpoint_epoch = max([int(re.sub('[^0-9]','', a)[2:]) for a in checkpoints])
last_checkpoint = sorted(checkpoints)[-1][:-5] + str(last_checkpoint_epoch) + ".pt"

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

cpu_device = torch.device("cpu")
print("CPU Device is ", cpu_device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is ", device)

# In[4] Load data (ex: wikipedia [user, item, timestamp])
print("Loading data")
path = os.getcwd()
raw_data = pd.read_csv(path + "/data/twitch100k.csv", sep=',', header=0)
print("Data looks like \n", raw_data.head())
# original ids to communities
streamer_community_dict = pd.Series(raw_data['community'].values, index=raw_data['streamer_name']).to_dict()
data = raw_data[['user_id', 'streamer_name', 'stop_time']]
print("Cleaned data looks like \n",data.head())
data = data.values
look_back = 50

# In[6]
unique_users = sorted(list(set(data[:, 0])))
unique_items = sorted(list(set(data[:, 1])))

# original ids to scaled ids
user_dic = {user:idx for (idx,user) in enumerate(unique_users)}
item_dic = {item:idx for (idx,item) in enumerate(unique_items)}

# scaled ids to original ids
reversed_user_dic = {value: key for key, value in user_dic.items()}
reversed_item_dic = {value: key for key, value in item_dic.items()}

# scaled id to communities
scaled_streamer_community_dict = {k:streamer_community_dict[k] for k, v in reversed_item_dic.items()}


for (idx, row) in enumerate(data):
    user,item,time = user_dic[row[0]],item_dic[row[1]],row[2]
    data[idx,0],data[idx,1] = int(user),int(item)

# print(data)
# In[5] Data preprocessing
print("Preprocessing Data")
original_data = train_test_split(data=data)
# Get data for the user
(train,test, train_items, test_items) = sequence_generator(original_data,look_back)



print("Train: {}, Test: {}".format(len(train),len(test)))

train_num,test_num = len(train),len(test)
train_labels,test_labels = [],[]


# In[]
# Cycle through all of the training points
train_labels = []
for i in range(len(train)):
    train_labels.append(train[i][1])
train = [train[i][0] for i in range(len(train))]



# In[]
print("Self Influence")

self_influence = approximate_tracin_batched(LSTM, sources=train, targets=train, source_labels=train_labels,
target_labels=train_labels, optimizer="SGD", paths=checkpoints, batch_size=2048, num_items=5673, device=device)


# In[]
print("Random Sample 1")
