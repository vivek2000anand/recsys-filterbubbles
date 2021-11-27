import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from os import listdir
from os.path import isfile, join
from tracin.tracin import save_tracin_checkpoint, load_tracin_checkpoint, calculate_tracin_influence
import pandas as pd
from LSTM_clean.utils import train_test_split, sequence_generator, get_diversity
from LSTM_clean.model import LSTM
import numpy as np
import re

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
# Subsetting Data for user 0
user0_data = np.array([o for o in original_data if o[0]==0])
# Get data for the user
(train,test, train_items, test_items) = sequence_generator(user0_data,look_back)
test_ground_truth = {i:test[i][1] for i in range(len(test))}

print("Train: {}, Test: {}".format(len(train),len(test)))

train_num,test_num = len(train),len(test)
train_labels,test_labels = [],[]

# Load the last checkpoint
curr_model = LSTM(input_size=128, output_size=5673, hidden_dim=64, n_layers=1) 
curr_model.LSTM.flatten_parameters()
optimizer = optim.SGD(curr_model.parameters(), lr=5e-2, momentum=0.9)
curr_model, optimizer, epoch, loss =load_tracin_checkpoint(curr_model, optimizer, last_checkpoint)

# Cycle through all of the training points
train_labels = []
for i in range(len(train)):
    # Get item embeddings
    train[i][0] = curr_model.item_emb(torch.LongTensor(train[i][0])).to(device)
    train_labels.append(train[i][1])
train_labels = torch.LongTensor(train_labels).to(device)
print("train is \n", train)
print("train_labels are \n", train_labels)
total_diversity = []
curr_model.to(device)
# Cycle through all of the training points
for iteration in range(int(train_num/64)+1):
    st_idx,ed_idx = iteration*64, (iteration+1)*64
    if ed_idx>train_num:
        ed_idx = train_num           
    optimizer.zero_grad()
    # Get previous item communities 
    prev_items = train_items[st_idx:ed_idx]
    prev_item_communities = [scaled_streamer_community_dict[item] for item in prev_items]
    output, hidden = curr_model.forward(torch.stack([train[i][0] for i in range(st_idx,ed_idx)],dim=0).to(device).detach())
    # Get top k values
    top10 = torch.topk(output, 10).indices.tolist()
    # Get communities for various items in the top 10 
    top10_communities =[]
    for instance in top10:
        top10_communities.append([])
        for item in instance:
            top10_communities[-1].append(scaled_streamer_community_dict[item])

    diversity = get_diversity(prev_item_communities, top10_communities)
    total_diversity = total_diversity + diversity

print("Net total diversity is ", str(sum(total_diversity)))
print("Total Filter Bubbles are ", str(sum([1 if i ==-1 else 0 for i in total_diversity ])))
print("Total Diverse points are ", str(sum([1 if i ==1 else 0 for i in total_diversity ])))
print("Total Moderate points are ", str(sum([1 if i ==0 else 0 for i in total_diversity ])))
        





# for i in range(train_num):
#     train[i][0] = model.item_emb(torch.LongTensor(train[i][0]).to(model.device))
#     train_labels.append(train[i][1])
# train_labels = torch.LongTensor(train_labels).to(model.device)
         
# for i in range(test_num):
#     test[i][0] = model.item_emb(torch.LongTensor(test[i][0]).to(model.device))
#     test_labels.append(test[i][1])
# test_labels = torch.LongTensor(test_labels).to(model.device)

# print("__________________________________________________________________________")


# source = torch.LongTensor(train[3][0]).to(device)
# source_label  = torch.LongTensor([train[3][1]]).to(device)

# target = torch.LongTensor(test[1][0]).to(device)
# target_label  = torch.LongTensor([test[1][1]]).to(device)

# print("Source is ", source)
# print("Source label is ", source_label)
# print("Target is ", target)
# print("Target Label is ", target_label)

# print("Calculating Influence \n __________________________________")
# influence = calculate_tracin_influence(LSTM, source, source_label, target, target_label, "SGD",  checkpoints)
# print("Influence is ", influence)

# print("__________________________________________________________________________")

# source = torch.LongTensor(train[100][0]).to(device)
# source_label  = torch.LongTensor([train[100][1]]).to(device)

# target = torch.LongTensor(test[5][0]).to(device)
# target_label  = torch.LongTensor([test[5][1]]).to(device)

# print("Source is ", source)
# print("Source label is ", source_label)
# print("Target is ", target)
# print("Target Label is ", target_label)

# print("Calculating Influence \n __________________________________")
# influence = calculate_tracin_influence(LSTM, source, source_label, target, target_label, "SGD",  checkpoints)
# print("Influence is ", influence)

# print("__________________________________________________________________________")


# source = torch.LongTensor(train[2][0]).to(device)
# source_label  = torch.LongTensor([train[2][1]]).to(device)

# target = torch.LongTensor(test[6][0]).to(device)
# target_label  = torch.LongTensor([test[6][1]]).to(device)

# print("Source is ", source)
# print("Source label is ", source_label)
# print("Target is ", target)
# print("Target Label is ", target_label)

# print("Calculating Influence \n __________________________________")
# influence = calculate_tracin_influence(LSTM, source, source_label, target, target_label, "SGD",  checkpoints)
# print("Influence is ", influence)

# print("__________________________________________________________________________")


# source = torch.LongTensor(train[209][0]).to(device)
# source_label  = torch.LongTensor([train[209][1]]).to(device)

# target = torch.LongTensor(test[4][0]).to(device)
# target_label  = torch.LongTensor([test[4][1]]).to(device)

# print("Source is ", source)
# print("Source label is ", source_label)
# print("Target is ", target)
# print("Target Label is ", target_label)

# print("Calculating Influence \n __________________________________")
# influence = calculate_tracin_influence(LSTM, source, source_label, target, target_label, "SGD",  checkpoints)
# print("Influence is ", influence)

# print("__________________________________________________________________________")

# %%
