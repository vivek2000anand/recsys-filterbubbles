'''
This is a supporting library for the loading the data.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from __future__ import division
import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import os, re
import argparse
from sklearn.preprocessing import scale
import pandas as pd 
# LOAD THE NETWORK
def load_network(data, time_scaling=True):
    '''
    This function loads the input network.

    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions. 
    '''

    user_sequence = []
    item_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None

    for i in range(data.shape[0]):
        user_sequence.append(int(data[i,0]))
        item_sequence.append(int(data[i,1]))
        if start_timestamp is None:
            start_timestamp = float(data[i,2])
        timestamp_sequence.append(float(data[i,2]) - start_timestamp) 
        feature_sequence.append([0])
        if data.shape[1]>=4:
            feature_sequence[i]= list(data[i,3:])

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]
    
    if time_scaling:
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)
    
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        feature_sequence]

