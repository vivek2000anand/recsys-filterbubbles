import torch
import torch.nn as nn
import numpy as np
import pandas as  pd
import copy
import time
import argparse
import os
from library_models import *
from library_data import *
from data_processing import *

def train_valid_test_generator(data, look_back):

    train,valid,test = [],[],[]
    unique_users = set(data[:,0])
    items_per_user = {int(user):[0 for i in range(look_back)] for user in unique_users}
    
    for (idx,row) in enumerate(data):
        user,item,time = int(row[0]),int(row[1]),row[2]
        items_per_user[user] = items_per_user[user][1:]+[item+1]
        current_items = items_per_user[user]
        if row[3]==0:
            train.append([current_items[:-1],current_items[-1]])                                                                                            
        elif row[3]==1:
            valid.append([current_items[:-1],current_items[-1]])
        else:
            test.append([current_items[:-1],current_items[-1]])
                                                                
    return train,valid,test


def main():
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help='name of the dataset')
    parser.add_argument('--gpu',default='0',type=str,help='GPU# will be used')
    parser.add_argument('--output',type=str, default = 'output/jodie.txt', help = "Which perturbation will be tested")
    parser.add_argument('--epochs', default=200, type = int, help='number of training epochs')
    parser.add_argument('--train_ratio', type=float, default="0.9",help='train ratio')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    raw_data = pd.read_csv(args.data_path, sep='\t', header=None)
    data = raw_data.values
    
    f = open(args.output,'w')

    look_back = 50

    unique_users = sorted(list(set(data[:, 0])))
    unique_items = sorted(list(set(data[:, 1])))
    user_dic = {user:idx for (idx,user) in enumerate(unique_users)}
    item_dic = {item:idx for (idx,item) in enumerate(unique_items)}
    for (idx, row) in enumerate(data):
        user,item,time = user_dic[row[0]],item_dic[row[1]],row[2]
        data[idx,0],data[idx,1] = int(user),int(item)
 
    MRR,HITS = [],[]
    for iteration in range(5): 
 
        new_data = filter_and_split(data,args.train_ratio)
        (train,valid,test) = train_valid_test_generator(new_data,look_back)
 
        model = LSTM(data = new_data,input_size=128, output_size=len(unique_items)+1, hidden_dim=128, n_layers=1, device=device).to(device)
    
        perf1 = model.traintest(train=train,valid=valid,test=test,epochs = args.epochs)
        MRR.append(perf1[0])
        HITS.append(perf1[1])
        print('MRR = {}\tRecall@10 = {}'.format(MRR[-1],HITS[-1]),file=f,flush=True)

    print(MRR,HITS)
    print('MRR_average = {}\tMRR_std_dev = {}'.format(np.average(MRR),np.std(MRR)))
    print('Recall@10_average = {}\tRecall@10_std_dev = {}'.format(np.average(HITS),np.std(HITS)))


if __name__ == "__main__":
    main()
