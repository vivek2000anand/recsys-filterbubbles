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
# Subsetting Data for user 0
user0_data = np.array([o for o in original_data if o[0]==0])
# Get data for the user
(train,test, train_items, test_items) = sequence_generator(user0_data,look_back)
test_ground_truth = {i:test[i][1] for i in range(len(test))}

print("Train: {}, Test: {}".format(len(train),len(test)))

train_num,test_num = len(train),len(test)
train_labels,test_labels = [],[]

# Load the last checkpoint
curr_model = LSTM(input_size=128, output_size=5673, hidden_dim=64, n_layers=1, device=device) 
curr_model.LSTM.flatten_parameters()
optimizer = optim.SGD(curr_model.parameters(), lr=5e-2, momentum=0.9)
curr_model, optimizer, epoch, loss =load_tracin_checkpoint(curr_model, optimizer, last_checkpoint)

# Cycle through all of the training points
train_labels = []
train_emb = [[-1] for _ in range(len(train))]
for i in range(len(train)):
    # Get item embeddings
    train_emb[i][0] = curr_model.item_emb(torch.LongTensor(train[i][0])).to(device)
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
    output, hidden = curr_model.forward(torch.stack([train_emb[i][0] for i in range(st_idx,ed_idx)],dim=0).to(device).detach())
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
        
# TracIn section Evaluating points against each other
filter_bubbles = []
filter_bubbles_labels = []
diverse_points = []
diverse_points_labels = []
moderate_points = []
moderate_points_labels = []
for i in range(len(total_diversity)):
    if total_diversity[i] == -1:
        # Filter bubble
        filter_bubbles.append(train[i][0])
        filter_bubbles_labels.append(train_labels[i])
    elif total_diversity[i] == 1:
        # Diverse Point
        diverse_points.append(train[i][0])
        diverse_points_labels.append(train_labels[i])
    else:
        # Neither, moderate point
        moderate_points.append(train[i][0])
        moderate_points_labels.append(train_labels[i])


# Dummy Experiment
# Compares Effects of filter bubbles on the diverse points
#influences = run_experiments(LSTM, sources=filter_bubbles, targets=diverse_points, sources_labels=filter_bubbles_labels,
#targets_labels=diverse_points_labels, paths=checkpoints, device=device)

influence = approximate_tracin_batched(LSTM, sources=filter_bubbles, targets=filter_bubbles, source_labels=filter_bubbles_labels,
target_labels=filter_bubbles_labels, optimizer="SGD", paths=checkpoints, batch_size=128, num_items=5673, device=device)
print("Influence is ", influence)