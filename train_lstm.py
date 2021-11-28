# In[1]
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import os

from tracin.tracin import save_tracin_checkpoint, calculate_tracin_influence, get_lr
from LSTM_clean.utils import train_test_split, sequence_generator
from LSTM_clean.model import LSTM
# In[3] Set CUDA env
epochs = 600

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='7'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)


# In[4] Load data (ex: wikipedia [user, item, timestamp])
print("Loading data")
path = os.getcwd()
raw_data = pd.read_csv(path + "/data/twitch100k.csv", sep=',', header=0)
print(raw_data.head())
streamer_community_dict = pd.Series(raw_data['community'].values, index=raw_data['streamer_name']).to_dict()
data = raw_data[['user_id', 'streamer_name', 'stop_time']]
print(data.head())
data = data.values
look_back = 50
print(data)

# In[6]
unique_users = sorted(list(set(data[:, 0])))
unique_items = sorted(list(set(data[:, 1])))

user_dic = {user:idx for (idx,user) in enumerate(unique_users)}
item_dic = {item:idx for (idx,item) in enumerate(unique_items)}

for (idx, row) in enumerate(data):
    user,item,time = user_dic[row[0]],item_dic[row[1]],row[2]
    data[idx,0],data[idx,1] = int(user),int(item)

print(data)
# In[5] Data preprocessing
print("Preprocessing Data")
original_data = train_test_split(data=data)
(train,test) = sequence_generator(original_data,look_back)
test_ground_truth = {i:test[i][1] for i in range(len(test))}

print("Train: {}, Test: {}".format(len(train),len(test)))

# %%
# In[6] Run LSTM
model = LSTM(input_size=128, output_size=len(unique_items)+1, hidden_dim=64, n_layers=1, device=device).to(device)
model.LSTM.flatten_parameters()
print("Model is ", model)
print("Training and testing")
original_prediction = model.traintest(train=train,test=test,epochs = epochs)
print("Finished")
# %%
