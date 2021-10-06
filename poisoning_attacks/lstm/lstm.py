import torch
import torch.nn as nn
import numpy as np
import pandas as  pd
import copy
import time
import argparse
import os
import rbo
from library_models import *
from collections import Counter

def filter_and_split(data=[]):

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
    
    test_len = 0
    for user in sequence_dic.keys():
        cur_test = int(0.1*len(sequence_dic[user]))
        for i in range(cur_test):
            interaction = sequence_dic[user].pop()
            new_data[interaction[0],3] = 1
        test_len += cur_test

    new_data = new_data[np.argsort(new_data[:,2]),:]
    print(data.shape,new_data.shape)
    return new_data,test_len

def train_test_generator(data, look_back,perturbed_users):

    train,test = [],[]
    unique_users = set(data[:,0])
    items_per_user = {int(user):[0 for i in range(look_back)] for user in unique_users}
    pert_dic = {int(user):1 for user in perturbed_users}
    
    for (idx,row) in enumerate(data):
        user,item,time = int(row[0]),int(row[1]),row[2]
        items_per_user[user] = items_per_user[user][1:]+[item+1]
        current_items = items_per_user[user]
        if row[3]==0:
            train.append([current_items[:-1],current_items[-1]])                                                                                            
        else:
            if user in pert_dic:
                test.append([current_items[:-1],current_items[-1],0])
            else:
                test.append([current_items[:-1],current_items[-1],1])
                                                                
    return train,test

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help='path of the dataset')
    parser.add_argument('--gpu',default='0',type=str,help='GPU# will be used')
    parser.add_argument('--attack_type',type=str, help = "Which attack will be tested")
    parser.add_argument('--attack_kind',type=str, help = "Deletion, Replacement, Injection attack")
    parser.add_argument('--output',type=str, default = 'lstm_output.txt', help = "Output file path")
    parser.add_argument('--epochs', default=50, type = int, help='number of training epochs')

    args = parser.parse_args()
    num_pert = 1
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
 
    output_path = args.output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    raw_data = pd.read_csv(args.data_path, sep='\t', header=None)
    data = raw_data.values[:,-3:]

    look_back = 50

    unique_users = sorted(list(set(data[:, 0])))
    unique_items = sorted(list(set(data[:, 1])))
    user_dic = {user:idx for (idx,user) in enumerate(unique_users)}
    item_dic = {item:idx for (idx,item) in enumerate(unique_items)}
    for (idx, row) in enumerate(data):
        user,item,time = user_dic[row[0]],item_dic[row[1]],row[2]
        data[idx,0],data[idx,1] = int(user),int(item)
 
    final_metrics = [[],[],[],[],[],[]]
    before_perf,after_perf = [[],[]],[[],[]]
    f = open(output_path,'w')
    for iteration in range(3): 
        perturbed_users = []
        (original_data,test_len) = filter_and_split(data=data)
        (train,test) = train_test_generator(original_data,look_back,perturbed_users)
        
        model = LSTM(data = original_data,input_size=128, output_size=len(unique_items)+1, hidden_dim=64, n_layers=1, device=device,seed=iteration).to(device)
        model.LSTM.flatten_parameters()
        print(model)
        original_model = copy.deepcopy(model)
        [original_probs,original_rank,temp, perf1] = model.traintest(train=train,test=test,epochs = args.epochs, original_probs=-1, original_rank=-1, final_metrics = [[]])
        
        if args.attack_type == 'cas':
            occurence_count = Counter(original_data[:,1])
            popular_item = occurence_count.most_common(1)[0][0] 
            least_popular_item = occurence_count.most_common()[-1][0] 
           
            cutoff = 50
            user_seq = {user:[0 for i in range(cutoff)] for user in set(original_data[:,0])}
            user_dic,item_dic = defaultdict(list),defaultdict(list)
            in_degree, num_child = np.zeros(original_data.shape[0]),np.zeros(original_data.shape[0])
            edges = defaultdict(list)
            count = 0
            for i in range(original_data.shape[0]):
                in_degree[i]=-1
                if original_data[i,3]==0:
                    count += 1
                    user,item = int(original_data[i,0]),int(original_data[i,1])
                    user_dic[user].append(i)
                    item_dic[item].append(i)
                    user_seq[user] = user_seq[user][1:]+[i]
                    in_degree[i] = 0

            valid = {}
            for user in user_seq:
                current_seq = user_seq[user]
                for i in range(cutoff):
                    valid[current_seq[i]]=1

            for user in user_dic.keys():
                cur_list = user_dic[user]
                for i in range(len(cur_list)-1):
                    j,k = cur_list[i],cur_list[i+1]
                    if j in valid and k in valid:
                        in_degree[k] += 1
                        edges[j].append(k)

            for item in item_dic.keys():
                cur_list = item_dic[item]
                for i in range(len(cur_list)-1):
                    j,k = cur_list[i],cur_list[i+1]
                    if j in valid and k in valid:
                        in_degree[k] += 1
                        edges[j].append(k)
            
            queue = []
            for i in range(original_data.shape[0]):
                if in_degree[i] == 0:
                    queue.append(i)

            while len(queue)!=0:
                root = queue.pop(0)
                check = np.zeros(original_data.shape[0])
                check[root]=1
                q2 = [root]
                count2 = 1
                while len(q2)!=0:
                    now = q2.pop(0)
                    for node in edges[now]:
                        if check[node]==0:
                            check[node]=1
                            q2.append(node)
                            count2+=1
                num_child[root] = count2

            chosen = np.argsort(num_child)[-num_pert:]
            if args.attack_kind=='deletion':
                tbd = []
                for idx in chosen:
                    maxv,maxp = num_child[idx],idx
                    user,item,time = int(original_data[maxp,0]),int(original_data[maxp,1]),original_data[maxp,2]
                    print('[CASSATA & Deletion] chosen interaction {}=({},{},{}) with cascading score {}'.format(maxp,user,item,time,maxv),file=f,flush=True)
                    tbd.append(maxp)
                    if user not in perturbed_users:
                        perturbed_users.append(user)
                new_data = np.delete(original_data,tbd,0)
            else:
                new_data = copy.deepcopy(original_data)
                for idx in chosen:
                    maxv,maxp = num_child[idx],idx
                    user,item,time = int(original_data[maxp,0]),int(original_data[maxp,1]),original_data[maxp,2]
                    print('[CASSATA & Replace] chosen interaction {}=({},{},{}) with cascading score {}'.format(maxp,user,item,time,maxv),file=f,flush=True)
                    replacement = int(least_popular_item)
                    new_data[maxp,1] = replacement
                    if user not in perturbed_users:
                        perturbed_users.append(user)

        elif args.attack_type=='random':
            candidates,items = [],[]
            early = {}
            for i in range(original_data.shape[0]):
                if original_data[i,3]==0:
                    candidates.append(i)
                    user = int(original_data[i,0])
                    if user not in early:
                        early[user] = i

            worst = np.random.choice(candidates,size = num_pert,replace=False)
#            worst = np.random.choice(list(early.values()),size=num_pert,replace=False)
 #           new_data = copy.deepcopy(original_data)
            tbd = []
            for idx in worst:
                tbd.append(idx)
                user,item,time = int(original_data[idx,0]),int(original_data[idx,1]),original_data[idx,2]
                print('Chosen interaction {}=({},{},{})'.format(idx,user,item,time),file=f,flush=True)
                new_item = np.random.choice(np.unique(original_data[:,1]))
                if user not in perturbed_users:
                    perturbed_users.append(user)
  #              new_data[idx,1] = new_item
            new_data = np.delete(original_data,tbd,0)
        else:
            candidates = {}
            for i in range(original_data.shape[0]):
                if original_data[i,3]==0:
                    user = int(original_data[i,0])
                    candidates[user] = i

            chosen = np.random.choice(list(candidates.values()),size = num_pert,replace=False)
            if args.attack_kind=='deletion':
                tbd = []
                for idx in chosen:
                    maxp = idx
                    user,item,time = int(original_data[maxp,0]),int(original_data[maxp,1]),original_data[maxp,2]
                    print('[last&random deletion] perturbed interaction {}=({},{},{})'.format(maxp,user,item,time),file=f,flush=True)
                    tbd.append(maxp)
                    if user not in perturbed_users:
                        perturbed_users.append(user)
                new_data = np.delete(original_data,tbd,0)
            else:
                new_data = copy.deepcopy(original_data)
                for idx in chosen:
                    maxp=idx
                    user,item,time = int(original_data[maxp,0]),int(original_data[maxp,1]),original_data[maxp,2]
                    print('[last&random replacement] perturbed interaction {}=({},{},{})'.format(maxp,user,item,time),file=f,flush=True)
                    new_data[maxp,1] = np.random.choice(list(set(original_data[:,1])))
                    if user not in perturbed_users:
                        perturbed_users.append(user)
         
        (train,test) = train_test_generator(new_data,look_back,perturbed_users) 
        model = copy.deepcopy(original_model)
        model.LSTM.flatten_parameters()
        [probs,rank,current_metrics,perf2] =  model.traintest(train=train,test=test, epochs = args.epochs,original_probs = original_probs, original_rank = original_rank, final_metrics =  [[],[],[],[],[],[]])
        print('\nMRR_diff\tHITS_diff\tRBO\tRank_diff\tProb_diff\tTop-K Jaccard\n',file = f,flush=True)
        
        for i in range(len(perf1)):
            before_perf[i].append(perf1[i])
            after_perf[i].append(perf2[i])

        for i in range(6):
            avg = np.average(current_metrics[i])
            med = np.median(current_metrics[i])
            std = np.std(current_metrics[i])
            final_metrics[i].append(avg)
            print('Avg = {}\tMed = {}\tStd = {}'.format(avg,med,std),file=f,flush=True)
        
        print('[Without perturbation] Avg MRR = {}\tAvg HITS@10 = {}'.format(np.average(before_perf[0]),np.average(before_perf[1])),file=f,flush=True)
        print('[With perturbation] Avg MRR = {}\tAvg HITS@10 = {}'.format(np.average(after_perf[0]),np.average(after_perf[1])),file=f,flush=True)

    for i in range(6):
        print(final_metrics[i],file=f,flush=True)

    for i in range(6):
        avg = np.average(final_metrics[i])
        print('({})'.format(avg),file=f,flush=True)

if __name__ == "__main__":
    main()
