import numpy as np
import pandas as pd
import copy

def filter_and_split(data=[],train_ratio=0.9):

    (users,counts) = np.unique(data[:,0],return_counts = True)
    
    users = users[counts>=10]

    sequence_dic =  {int(user):[] for user in set(data[:,0])}
    
    user_dic = {int(user):idx for (idx,user) in enumerate(users)}
    new_data = []
    for i in range(data.shape[0]):
        if int(data[i,0]) in user_dic:
            new_data.append([int(data[i,0]),int(data[i,1]),data[i,2],0])

    new_data = np.array(new_data)
    new_data = new_data[np.argsort(new_data[:,2]),:]

    train_cutoff = int(train_ratio*new_data.shape[0])
    valid_cutoff = int(train_ratio*train_cutoff)
    for i in range(valid_cutoff,train_cutoff):
        new_data[i,3]=1
    for i in range(train_cutoff,new_data.shape[0]):
        new_data[i,3]=2

    print(data.shape,new_data.shape)
    return new_data



