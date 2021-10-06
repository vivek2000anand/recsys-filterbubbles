'''
This is a supporting library with the code of the model.
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
import time
import random
import rbo
from scipy import stats
from library_data import *
from tqdm import trange
import library_models as lib
import copy

total_reinitialization_count=0

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

# LATENT CROSS
class ltcross(nn.Module):
    def __init__(self, embedding_dim, num_features, num_users, num_items,seed):
        super(ltcross, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_items = num_items
        self.last_size = self.embedding_dim // 2 + num_features + 1
        self.random_seed = seed

        self.initial_user_embedding = nn.Parameter(F.normalize(torch.rand(embedding_dim), dim=0))
       
        self.user_lstm = nn.LSTMCell(num_items + num_features + 1, self.embedding_dim // 2)
        
        self.linear9 = nn.Linear(self.embedding_dim // 2 + num_features + 1, num_items)
        self.embed_layer1 = NormalLinear(1 + num_features, num_items + 1 + num_features)
        self.embed_layer2 = NormalLinear(1 + num_features, self.embedding_dim // 2 + 1 + num_features)

    def forward(self, user_embeddings, item_embeddings, user_timediffs=None, features=None, time_seconds=None, prev_time_seconds=None):
        inp2 = torch.cat([item_embeddings, features, user_timediffs], dim=1)
        inp2 *= (1 + self.embed_layer1(torch.cat([user_timediffs, features], dim=1)))

        h0 = user_embeddings[:, :self.embedding_dim // 2]
        c0 = user_embeddings[:, self.embedding_dim // 2:]

        h1, c1 = self.user_lstm(inp2, (h0,c0))

        user_embedding_output = torch.cat([h1,c1], dim=1)
        return F.normalize(user_embedding_output)


    def predict_current_item(self, user_embeddings, user_timediffs=None, features=None, user_embeddings_static=None):
        inp1 = torch.cat([user_embeddings[:, :self.embedding_dim // 2], user_timediffs, features], dim=1)
        inp1 *= (1 + self.embed_layer2(torch.cat([user_timediffs, features], dim=1)))
        self.intermediate = inp1
        X_out = self.linear9(inp1)
        return X_out

    def traintest(self,data, original_probs, original_rank, final_metrics, perturbed_users,test_len,epochs,device):
        
        [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
         item2id, item_sequence_id, item_timediffs_sequence, 
         timestamp_sequence, feature_sequence] = load_network(data)

        num_interactions = len(user_sequence_id)
        num_users = len(user2id) 
        num_items = len(item2id)+1 # one extra item for "none-of-these"
        num_features = len(feature_sequence[0])
        embedding_dim = self.embedding_dim
        print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n\n" % (num_users, num_items, num_interactions))
        
        train_end_idx = int(num_interactions)-test_len
        test_end_idx = int(num_interactions)

        # INITIALIZE MODEL AND PARAMETERS
        crossEntropyLoss = nn.CrossEntropyLoss()

        # initialize embedding
        initial_item_embedding = torch.eye(num_items).to(device)
        user_embeddings_static = torch.eye(num_users).to(device) # one-hot vectors for static embeddings

        learning_rate = 0.001
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        perturbed_dic = {}
        for user in perturbed_users:
            perturbed_dic[user] = 1
        
        predicted_rank = [0 for i in range(test_len)]
        probs = [0 for i in range(test_len)]

        MRR,HITS,testidx,test_count,tcount = 0,0,0,0,0
 
        print("*** Training the LatentCross model for %d epochs ***" % epochs)

        self.inter = torch.zeros(int(epochs/10), num_interactions,self.last_size)
        self.inter2 = torch.zeros(int(epochs/10), num_interactions,num_items)
         
        with trange(epochs) as progress_bar1:
            for ep in progress_bar1:
                progress_bar1.set_description('Epoch %d of %d' % (ep, epochs))

                # REINITIALIZE USER EMBEDDINGS
                user_embeddings = self.initial_user_embedding.repeat(num_users, 1)
                item_embeddings = initial_item_embedding.clone()

                optimizer.zero_grad()
                reinitialize_tbatches()
                total_loss, loss, total_interaction_count,tcount = 0, 0, 0,0

                tbatch_start_time = None
                tbatch_to_insert = -1
                tbatch_full = False

                # TRAIN TILL THE END OF TRAINING INTERACTION IDX 
                with trange(num_interactions) as progress_bar2: 
                    for j in progress_bar2:
                        
                        progress_bar2.set_description('Processed %dth interactions' % j)

                        if data[j,3]==0:
                            # READ INTERACTION J
                            userid = user_sequence_id[j]
                            itemid = item_sequence_id[j]
                            feature = feature_sequence[j]
                            user_timediff = user_timediffs_sequence[j]

                            # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                            tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1 
                            lib.tbatchid_user[userid] = tbatch_to_insert 
                            lib.tbatchid_item[itemid] = tbatch_to_insert

                            lib.current_tbatches_user[tbatch_to_insert].append(userid)
                            lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                            lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                            lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                            lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                            tcount += 1

                        timestamp = timestamp_sequence[j]
                        if tbatch_start_time is None:
                            tbatch_start_time = timestamp

                        # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
                        if (tcount>=1024 or j==num_interactions-1) and tbatch_to_insert!=-1:
                            tcount = 0
                            tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES
                            
                            inters,indices = [],[]
                            for i in range(len(lib.current_tbatches_user)):
                                   
                                total_interaction_count += len(lib.current_tbatches_interactionids[i])

                                # LOAD THE CURRENT TBATCH
                                tbatch_userids = torch.LongTensor(lib.current_tbatches_user[i]).to(device) # Recall "lib.current_tbatches_user[i]" has unique elements
                                tbatch_itemids = torch.LongTensor(lib.current_tbatches_item[i]).to(device) # Recall "lib.current_tbatches_item[i]" has unique elements
                                tbatch_interactionids = torch.LongTensor(lib.current_tbatches_interactionids[i])
                                feature_tensor = torch.Tensor(lib.current_tbatches_feature[i]).to(device) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tenso
                                user_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_user_timediffs[i]).to(device)).unsqueeze(1)
                                # PROJECT USER EMBEDDING TO CURRENT TIME
                                user_embedding_input = user_embeddings[tbatch_userids,:]
                                item_embedding_input = item_embeddings[tbatch_itemids,:]
                                user_embedding_output = self.forward(user_embedding_input, item_embedding_input, features=feature_tensor, user_timediffs=user_timediffs_tensor)
                                user_embeddings[tbatch_userids,:] = user_embedding_output 
                               
                                curr_item_pred = self.predict_current_item(user_embedding_input, user_timediffs_tensor, feature_tensor)
 
                                if (ep+1)%10==0:               
                                    target = torch.zeros((len(tbatch_interactionids),num_items))
                                    current_items = lib.current_tbatches_item[i]
                                    for (idx,item) in enumerate(current_items):
                                        target[idx,item]=1
                                    target = target.to(device)
                                    self.inter2[int((ep+1)/10)-1,tbatch_interactionids,:] = (curr_item_pred-target).detach().cpu().clone()
                                    self.inter[int((ep+1)/10)-1,tbatch_interactionids,:] = self.intermediate.detach().cpu().clone()

                                loss += crossEntropyLoss(curr_item_pred, Variable(tbatch_itemids))

                            # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                            total_loss += loss.item()
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                            # RESET LOSS FOR NEXT T-BATCH
                            loss = 0
                            item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                            user_embeddings.detach_()
                            
                            # REINITIALIZE
                            reinitialize_tbatches()
                            tbatch_to_insert = -1

                # END OF ONE EPOCH 
                print("\n\nTotal loss in this epoch = %f" % (total_loss))

        reinitialize_tbatches()
        total_loss, loss, total_interaction_count = 0, 0, 0

        optimizer.zero_grad()
        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        MRR,HITS,testidx,test_count,tcount = 0,0,0,0,0
        # TRAIN TILL THE END OF TRAINING INTERACTION IDX 
        with trange(num_interactions) as progress_bar2: 
            for j in progress_bar2:
                progress_bar2.set_description('Processed %dth interactions' % j)
                
                if data[j,3]==1:
                
                    # READ INTERACTION J
                    userid = user_sequence_id[j]
                    itemid = item_sequence_id[j]
                    feature = feature_sequence[j]
                    user_timediff = user_timediffs_sequence[j]

                    # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                    tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1 
                    lib.tbatchid_user[userid] = tbatch_to_insert 
                    lib.tbatchid_item[itemid] = tbatch_to_insert

                    lib.current_tbatches_user[tbatch_to_insert].append(userid)
                    lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                    lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                    lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                    lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                    lib.current_tbatches_testids[tbatch_to_insert].append(testidx)
                    testidx += 1
                    tcount += 1

                timestamp = timestamp_sequence[j]
                if tbatch_start_time is None:
                    tbatch_start_time = timestamp

                # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
 
                if (tcount>=1024 or j==num_interactions-1) and tbatch_to_insert!=-1:
                    tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES
                    tcount= 0
                    
                    inters,indices = [],[]
                    for i in range(len(lib.current_tbatches_user)):
                           
                        total_interaction_count += len(lib.current_tbatches_interactionids[i])

                        # LOAD THE CURRENT TBATCH
                        tbatch_userids = torch.LongTensor(lib.current_tbatches_user[i]).to(device) # Recall "lib.current_tbatches_user[i]" has unique elements
                        tbatch_itemids = torch.LongTensor(lib.current_tbatches_item[i]).to(device) # Recall "lib.current_tbatches_item[i]" has unique elements
                        tbatch_testids = lib.current_tbatches_testids[i]
                        tbatch_interactionids = torch.LongTensor(lib.current_tbatches_interactionids[i])
                        feature_tensor = Variable(torch.Tensor(lib.current_tbatches_feature[i]).to(device)) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                        user_timediffs_tensor = torch.Tensor(lib.current_tbatches_user_timediffs[i]).to(device).unsqueeze(1)

                        # PROJECT USER EMBEDDING TO CURRENT TIME
                        user_embedding_input = user_embeddings[tbatch_userids,:]
                        item_embedding_input = item_embeddings[tbatch_itemids,:]

                        # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION

                        user_embedding_output = self.forward(user_embedding_input, item_embedding_input, features=feature_tensor, user_timediffs=user_timediffs_tensor)
                        user_embeddings[tbatch_userids,:] = user_embedding_output 
                        curr_item_pred = self.predict_current_item(user_embedding_input, user_timediffs_tensor, feature_tensor)
                        loss += crossEntropyLoss(curr_item_pred, Variable(tbatch_itemids)) 

                        softmax = nn.Softmax(dim=1).to(device)
                        current_val = np.zeros((len(tbatch_interactionids),1))
                        prob = softmax(curr_item_pred).data.cpu().numpy()
                        original_prob = copy.deepcopy(prob)
                        for (idx,tidx) in enumerate(tbatch_testids):
                            user = tbatch_userids[idx]
                            item = tbatch_itemids[idx]
                            current_val[idx,0] = prob[idx,item]
                            if original_probs!=-1:
                                original_prob[idx,:] = original_probs[tidx]

                        new_prob = prob - current_val
                        ranks = np.count_nonzero(new_prob>0,axis=1)
                       
                        if original_probs!=-1:
                            rank1,rank2 = np.argsort(original_prob,axis=1)[:,::-1],np.argsort(prob,axis=1)[:,::-1]

                        for (idx,tidx) in enumerate(tbatch_testids): 
                            predicted_rank[tidx] = ranks[idx]+1 
                            probs[tidx] = prob[idx,:]
                            user = tbatch_userids[idx]
                            item = tbatch_itemids[idx]
                            
                            test_count += 1
                            MRR += (1/predicted_rank[tidx])
                            HITS += (1 if predicted_rank[tidx]<=10 else 0)
                            if user not in perturbed_dic and original_probs!=-1:
                                MRR_diff = abs((1/predicted_rank[tidx])-(1/original_rank[tidx]))
                                HITS_diff = abs((1 if predicted_rank[tidx]<=10 else 0)-(1 if original_rank[tidx]<=10 else 0))
                                RBO = rbo.RankingSimilarity(rank1[idx,:], rank2[idx,:]).rbo()
                                rank_diff = abs(predicted_rank[tidx]-original_rank[tidx])
                                prob_diff = abs(probs[tidx][item] - original_probs[tidx][item])
                                jaccard = np.intersect1d(rank1[idx,:10],rank2[idx,:10]).shape[0]/np.union1d(rank1[idx,:10],rank2[idx,:10]).shape[0]
                                final_metrics[0].append(MRR_diff)
                                final_metrics[1].append(HITS_diff)
                                final_metrics[2].append(RBO)
                                final_metrics[3].append(rank_diff)
                                final_metrics[4].append(prob_diff) 
                                final_metrics[5].append(jaccard)
            
                    # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # RESET LOSS FOR NEXT T-BATCH
                    loss = 0
                    item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                    user_embeddings.detach_()
                    
                    # REINITIALIZE
                    reinitialize_tbatches()
                    tbatch_to_insert = -1
        

        MRR /= test_count
        HITS /= test_count
        print("\n\n*** Testing complete. MRR = {}\tHITS = {}\tTest len = {}\tActual test count = {} ***\n\n".format(MRR,HITS,test_len,test_count))
        return [probs,predicted_rank,final_metrics,[MRR,HITS]]

# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item,current_tbatches_testids
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_testids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1

