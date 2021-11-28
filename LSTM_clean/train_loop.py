import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import os

from model import LSTM

def train_test_split(data=[]):
    
    """\
    Description:
    ------------
        Sort by each user's interation. Tag train & test & (valid) set
    """

    (users,counts) = np.unique(data[:,0],return_counts = True)
    users = users[counts>=10]

    sequence_dic,pert_dic =  {int(user):[] for user in set(data[:,0])}, {int(user):[] for user in set(data[:,0])}
    
    user_dic = {int(user):idx for (idx,user) in enumerate(users)}
    new_data = []
    for i in range(data.shape[0]):
        if int(data[i,0]) in user_dic:
            new_data.append([int(data[i,0]),int(data[i,1]),data[i,2],0])

    new_data = np.array(new_data)

    for i in range(new_data.shape[0]):
        sequence_dic[int(new_data[i,0])].append([i,int(new_data[i,1]),new_data[i,2]])
    
    for user in sequence_dic.keys():
        cur_test = int(0.1*len(sequence_dic[user]))
        for i in range(cur_test):
            interaction = sequence_dic[user].pop()
            new_data[interaction[0],3] = 2

        #cur_val = int(0.1*len(sequence_dic[user]))
        #for i in range(cur_val):
        #    interaction = sequence_dic[user].pop()
        #    new_data[interaction[0],3] = 1

    return new_data

def sequence_generator(data, look_back = 50):

    """\
    Description:
    ------------
        Input data for LSTM: Convert to user trajectory (maximum length: look back)
    """

    train,test, valid = [],[],[]
    unique_users = set(data[:,0])
    items_per_user = {int(user):[0 for i in range(look_back)] for user in unique_users}
    
    for (idx,row) in enumerate(data):
      user,item,time = int(row[0]),int(row[1]),row[2]
      items_per_user[user] = items_per_user[user][1:]+[item+1]
      current_items = items_per_user[user]
      if row[3]==0:
        train.append([current_items[:-1],current_items[-1]])
      elif row[3]==2:
        test.append([current_items[:-1],current_items[-1]])
      else:
        valid.append([current_items[:-1],current_items[-1]])
                                                                
    return train,test #,valid


epochs = 1000

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)


print("Loading data")
path = "../data/"
raw_data = pd.read_csv(path + "wikipedia.tsv", sep='\t', header=None)
data = raw_data.values[:,-3:]
look_back = 50

unique_users = sorted(list(set(data[:, 0])))
unique_items = sorted(list(set(data[:, 1])))
user_dic = {user:idx for (idx,user) in enumerate(unique_users)}
item_dic = {item:idx for (idx,item) in enumerate(unique_items)}

for (idx, row) in enumerate(data):
    user,item,time = user_dic[row[0]],item_dic[row[1]],row[2]
    data[idx,0],data[idx,1] = int(user),int(item)


# In[5] Data preprocessing
print("Preprocessing Data")
original_data = train_test_split(data=data)
(train,test) = sequence_generator(original_data,look_back)
test_ground_truth = {i:test[i][1] for i in range(len(test))}

print("Train: {}, Test: {}".format(len(train),len(test)))

# In[6] Run LSTM
model = LSTM(input_size=128, output_size=len(unique_items)+1, hidden_dim=64, n_layers=1, device=device).to(device)
model.LSTM.flatten_parameters()
print("Model is ", model)
print("Training and testing")
original_prediction = model.traintest(train=train,test=test,epochs = epochs)
print("Finished")

# %%