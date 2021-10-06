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


# THE JODIE MODULE
class JODIE(nn.Module):
    def __init__(self, embedding_dim, num_features, num_users, num_items):
        super(JODIE, self).__init__()

        print("*** Initializing the JODIE model ***")
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items
        self.last_size =  self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2
        self.last_size2 = self.item_static_embedding_size + self.embedding_dim
        # INITIALIZE EMBEDDING
        self.initial_user_embedding = nn.Parameter(F.normalize(torch.rand(embedding_dim), dim=0))  # the initial user and item embeddings are learned during training as well
        self.initial_item_embedding = nn.Parameter(F.normalize(torch.rand(embedding_dim), dim=0))


        rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1 + num_features

        print("Initializing user and item RNNs")
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)

        print("Initializing linear layers")
        self.prediction_layer = nn.Linear(self.last_size,self.last_size2)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        print("*** JODIE initialization complete ***\n\n")

    def forward(self, user_embeddings, item_embeddings, timediffs=None, features=None, select=None):
        if select == 'item_update':
            input1 = torch.cat([user_embeddings, timediffs, features], dim=1).requires_grad_()
            item_embedding_output = self.item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            input2 = torch.cat([item_embeddings, timediffs, features], dim=1).requires_grad_()
            user_embedding_output = self.user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)

        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            # user_projected_embedding = torch.cat([input3, item_embeddings], dim=1)
            return user_projected_embedding

    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out

    def traintest(self,data, original_probs, original_rank, final_metrics, perturbed_users,test_len,epochs,device):
        [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
         item2id, item_sequence_id, item_timediffs_sequence,
         timestamp_sequence, feature_sequence] = load_network(data)

        num_interactions = len(user_sequence_id)
        num_users = len(user2id)
        num_items = len(item2id) + 1  # one extra item for "none-of-these"
        num_features = len(feature_sequence[0])
        print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n\n" % (num_users, num_items, num_interactions))

        # INITIALIZE MODEL AND PARAMETERS
        embedding_dim = self.embedding_dim
        crossEntropyLoss = nn.CrossEntropyLoss()
        MSELoss = nn.MSELoss()

        item_embedding_static = torch.eye(num_items).to(device)  # one-hot vectors for static embeddings
        user_embedding_static = torch.eye(num_users).to(device)  # one-hot vectors for static embeddings

        # INITIALIZE MODEL
        learning_rate = 1e-3
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        # RUN THE JODIE MODEL
        '''
        THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
        '''
        print("*** Training the JODIE model for %d epochs ***" % epochs)

        perturbed_dic = {}
        for user in perturbed_users:
            perturbed_dic[user] = 1

        predicted_rank = [0 for i in range(test_len)]
        probs = [0 for i in range(test_len)]

        MRR, HITS, testidx, test_count, tcount = 0, 0, 0, 0, 0

        with trange(epochs) as progress_bar1:
            for ep in progress_bar1:
                progress_bar1.set_description('Epoch %d of %d' % (ep, epochs))

                epoch_start_time = time.time()
                user_embeddings = self.initial_user_embedding.repeat(num_users, 1)  # initialize all users to the same embedding
                item_embeddings = self.initial_item_embedding.repeat(num_items, 1)  # initialize all items to the same embedding

                optimizer.zero_grad()
                reinitialize_tbatches()
                total_loss, loss, total_interaction_count = 0, 0, 0

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
                            item_timediff = item_timediffs_sequence[j]

                            # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                            tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                            lib.tbatchid_user[userid] = tbatch_to_insert
                            lib.tbatchid_item[itemid] = tbatch_to_insert

                            lib.current_tbatches_user[tbatch_to_insert].append(userid)
                            lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                            lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                            lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                            lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                            lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                            lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])
                            tcount+=1
                        
                        timestamp = timestamp_sequence[j]

                        # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
                        if (tcount >= 1024 or j == num_interactions - 1) and tbatch_to_insert != -1:
                            tcount = 0
                           
                            for i in range(len(lib.current_tbatches_user)):

                                total_interaction_count += len(lib.current_tbatches_interactionids[i])

                                tbatch_userids = torch.LongTensor(lib.current_tbatches_user[i]).to(device)  # Recall "lib.current_tbatches_user[i]" has unique elements
                                tbatch_itemids = torch.LongTensor(lib.current_tbatches_item[i]).to(device)  # Recall "lib.current_tbatches_item[i]" has unique elements
                                tbatch_interactionids = torch.LongTensor(lib.current_tbatches_interactionids[i])
                                feature_tensor = torch.Tensor(lib.current_tbatches_feature[i]).to(device)  # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                                user_timediffs_tensor = torch.Tensor(lib.current_tbatches_user_timediffs[i]).to(device).unsqueeze(1)
                                item_timediffs_tensor = torch.Tensor(lib.current_tbatches_item_timediffs[i]).to(device).unsqueeze(1)
                                tbatch_itemids_previous = torch.LongTensor(lib.current_tbatches_previous_item[i]).to(device)
                                item_embedding_previous = item_embeddings[tbatch_itemids_previous, :]

                                # PROJECT USER EMBEDDING TO CURRENT TIME
                                user_embedding_input = user_embeddings[tbatch_userids, :]
                                user_projected_embedding = self.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                                user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous, :], user_embedding_static[tbatch_userids, :]], dim=1)

                                # PREDICT NEXT ITEM EMBEDDING
                                predicted_item_embedding = self.predict_item_embedding(user_item_embedding)

                                # CALCULATE PREDICTION LOSS
                                item_embedding_input = item_embeddings[tbatch_itemids, :]

                                target = torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids, :]], dim=1).detach()
                                loss += MSELoss(predicted_item_embedding, target)
                                # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                                user_embedding_output = self.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                                item_embedding_output = self.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                                item_embeddings[tbatch_itemids, :] = item_embedding_output
                                user_embeddings[tbatch_userids, :] = user_embedding_output

                                # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                                loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                                loss += MSELoss(user_embedding_output, user_embedding_input.detach())


                            # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                            total_loss += loss.item()
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                            # RESET LOSS FOR NEXT T-BATCH
                            loss = 0
                            item_embeddings.detach_()  # Detachment is needed to prevent double propagation of gradient
                            user_embeddings.detach_()

                            reinitialize_tbatches()
                            tbatch_to_insert = -1

                print("Last epoch took {} minutes".format((time.time() - epoch_start_time) / 60))
                # END OF ONE EPOCH
                print("\n\nTotal loss in this epoch = %f" % (total_loss))

        tbatch_start_time = None
        loss = 0
        # FORWARD PASS
        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count = 0, 0, 0

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False
                        
        if original_probs!=-1:
            ranks = np.argsort(original_probs, axis=1)


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
                    item_timediff = item_timediffs_sequence[j]

                    # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                    tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                    lib.tbatchid_user[userid] = tbatch_to_insert
                    lib.tbatchid_item[itemid] = tbatch_to_insert

                    lib.current_tbatches_user[tbatch_to_insert].append(userid)
                    lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                    lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                    lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                    lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                    lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                    lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])
                    lib.current_tbatches_testids[tbatch_to_insert].append(testidx)
                    testidx += 1
                    tcount += 1

                timestamp = timestamp_sequence[j]

                # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
                if (tcount>=1024 or j==num_interactions-1) and tbatch_to_insert!=-1:
                    tcount=0

                    for i in range(len(lib.current_tbatches_user)):

                        total_interaction_count += len(lib.current_tbatches_interactionids[i])

                        lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).to(device)
                        lib.current_tbatches_item[i] = torch.LongTensor(lib.current_tbatches_item[i]).to(device)
                        lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).to(device)
                        lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i]).to(device)

                        lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).to(device)
                        lib.current_tbatches_item_timediffs[i] = torch.Tensor(lib.current_tbatches_item_timediffs[i]).to(device)
                        lib.current_tbatches_previous_item[i] = torch.LongTensor(lib.current_tbatches_previous_item[i]).to(device)

                        tbatch_userids = lib.current_tbatches_user[i]  # Recall "lib.current_tbatches_user[i]" has unique elements
                        tbatch_itemids = lib.current_tbatches_item[i]  # Recall "lib.current_tbatches_item[i]" has unique elements
                        tbatch_interactionids = lib.current_tbatches_interactionids[i]
                        feature_tensor = Variable(lib.current_tbatches_feature[i])  # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                        tbatch_testids = lib.current_tbatches_testids[i]

                        user_timediffs_tensor = lib.current_tbatches_user_timediffs[i].unsqueeze(1)
                        item_timediffs_tensor = lib.current_tbatches_item_timediffs[i].unsqueeze(1)
                        tbatch_itemids_previous = lib.current_tbatches_previous_item[i]
                        item_embedding_previous = item_embeddings[tbatch_itemids_previous, :]

                        # PROJECT USER EMBEDDING TO CURRENT TIME
                        user_embedding_input = user_embeddings[tbatch_userids, :]
                        user_projected_embedding = self.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous, :], user_embedding_static[tbatch_userids, :]], dim=1)

                        # PREDICT NEXT ITEM EMBEDDING
                        predicted_item_embedding = self.predict_item_embedding(user_item_embedding)

                        i_emb = torch.cat([item_embeddings, item_embedding_static], dim=1)
                        for (idx, tidx) in enumerate(tbatch_testids):
                            # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS
                            euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding[idx].repeat(num_items, 1), i_emb).squeeze(-1)
                            probs[tidx] = euclidean_distances.data.cpu().numpy()

                            user = tbatch_userids[idx]
                            item = tbatch_itemids[idx]
                            # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
                            true_item_distance = euclidean_distances[tbatch_itemids[idx]]
                            euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
                            true_item_rank = np.sum(euclidean_distances_smaller) + 1
                            predicted_rank[tidx] = true_item_rank
 
                            test_count += 1
                            MRR += (1 / predicted_rank[tidx])
                            HITS += (1 if predicted_rank[tidx] <= 10 else 0)
                           
                            if original_probs!=-1:
                                rank1, rank2 = ranks[tidx], np.argsort(probs[tidx])
                                if user not in perturbed_dic:
                                    MRR_diff = abs((1 / predicted_rank[tidx]) - (1 / original_rank[tidx]))
                                    HITS_diff = abs((1 if predicted_rank[tidx] <= 10 else 0) - (1 if original_rank[tidx] <= 10 else 0))
                                    RBO = rbo.RankingSimilarity(rank1,rank2).rbo()
                                    rank_diff = abs(predicted_rank[tidx] - original_rank[tidx])
                                    prob_diff = abs(probs[tidx][item] - original_probs[tidx][item])
                                    jaccard = np.intersect1d(rank1[:10],rank2[:10]).shape[0]/np.union1d(rank1[:10],rank2[:10]).shape[0]
                                    final_metrics[0].append(MRR_diff)
                                    final_metrics[1].append(HITS_diff)
                                    final_metrics[2].append(RBO)
                                    final_metrics[3].append(rank_diff)
                                    final_metrics[4].append(prob_diff)
                                    final_metrics[5].append(jaccard)

                                # CALCULATE PREDICTION LOSS
                        item_embedding_input = item_embeddings[tbatch_itemids, :]
                        loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids, :]], dim=1).detach())

                        # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                        user_embedding_output = self.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                        item_embedding_output = self.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                        item_embeddings[tbatch_itemids, :] = item_embedding_output
                        user_embeddings[tbatch_userids, :] = user_embedding_output

                        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                        loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                        loss += MSELoss(user_embedding_output, user_embedding_input.detach())

                    # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # RESET LOSS FOR NEXT T-BATCH
                    loss = 0
                    item_embeddings.detach_()  # Detachment is needed to prevent double propagation of gradient
                    user_embeddings.detach_()

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

