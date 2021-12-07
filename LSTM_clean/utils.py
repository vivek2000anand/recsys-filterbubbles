"""This file should probably be refactored

Contains a lot of utils for manipulating data and preparing data for training

Also contains utils for analysis"""

from collections import Counter, defaultdict
import pickle
import numpy as np
import math
from functools import cache


def printl(length=80):
    """Prents a horizontal line for prettier debugging"""
    print(length * "-")


def filter_and_split_data(
    data, min_items_per_user=10, train_cutoff=0.8, valid_cutoff=0.9
):
    """Takes in NX3 array and returns sequence-like data

    The data argument should be formatted as:
        Col 1: User ID
        Col 2: Item ID
        Col 3: Time

    Args:
        min_items_per_user: Only keep users with >= this # number of interactions
        train_cutoff: [0, train_cutoff] will be labeled as training data
        valid_cutoff: (train_cutoff, valid_cutoff] will be labeled as validation data
                    : (valid_cutoff, 1.0] will be labeled as testing data

    """
    # 1. Only keep users with >= threshold items
    users, counts = np.unique(data[:, 0], return_counts=True)
    total_items_per_user = {
        user: count for user, count in zip(users, counts) if count >= min_items_per_user
    }

    # 2. Sort our user data by time
    user_to_sequence = defaultdict(list)
    for user, item, time in data:
        user_to_sequence[user].append((time, item))

    for val in user_to_sequence.values():
        val.sort()

    new_data = []
    for user, sequence in user_to_sequence.items():
        for time, item in sequence:
            new_data.append([user, item, time])
    data = new_data

    # 3. Splitting into train, valid, test
    count_per_user = defaultdict(int)
    latest_time_per_user = defaultdict(int)
    new_data = []

    for user, item, time in data:
        user, item, time = int(user), int(item), int(time)
        if user not in total_items_per_user:
            continue
        if time < latest_time_per_user[user]:
            raise ValueError("Your data by user is not sorted by time!")

        total_items = total_items_per_user[user]
        latest_time_per_user[user] = time
        count_per_user[user] += 1

        # Version where we use 1 datapoint for validation, 1 datapoint for test
        # if count_per_user[user] <= total_items - 2:
        #     new_data.append([user, item, time, 0])
        # elif count_per_user[user] == total_items - 1:
        #     new_data.append([user, item, time, 1])
        # elif count_per_user[user] == total_items:
        #     new_data.append([user, item, time, 2])
        # else:
        #     raise ValueError(
        #         "There's a bug in the counting of total items or curr items!"
        #     )

        # Original Version
        if count_per_user[user] <= train_cutoff * total_items:
            new_data.append([user, item, time, 0])
        elif count_per_user[user] <= valid_cutoff * total_items:
            new_data.append([user, item, time, 1])
        elif count_per_user[user] <= total_items:
            new_data.append([user, item, time, 2])
        else:
            raise ValueError(
                "There's a bug in the counting of total items or curr items!"
            )

    return np.array(new_data)


# NOTE: Sequence generator already increases item ids by 1!
def sequence_generator(data, look_back=50):
    """Takes in data and converts to user trajectory"""

    train, valid, test = [], [], []
    unique_users = set(data[:, 0])
    items_per_user = {int(user): [0 for _ in range(look_back)] for user in unique_users}

    for user, item, time, split in data:
        # NOTE: Item ID increase happens here
        items_per_user[user] = items_per_user[user][1:] + [item + 1]
        # items_per_user[user] = items_per_user[user][1:] + [item]
        current_items = items_per_user[user]
        if split == 0:
            train.append([current_items[:-1], current_items[-1]])
        elif split == 1:
            valid.append([current_items[:-1], current_items[-1]])
        elif split == 2:
            test.append([current_items[:-1], current_items[-1]])
        else:
            raise ValueError(
                "Some of the data has not been split into train/valid/test!"
            )

    return train, valid, test


def reindex_and_save_communities(
    train_data,
    valid_data,
    test_data,
    original_df,
    item_key="streamer_name",
    community_key="community",
):
    """Fills in gaps between item ids to match the LSTM indices, saves a community dict using the original df

    NOTE: The data has already been incremented by 1 from the sequence_generator due to 0 padding
          This means that the mapping back to the df itemids needs to fill in the gaps and subtract 1
    """
    # NOTE: it is very important that this is a reference because we use all_data to reindex the elements in train, valid, test
    all_data = train_data + valid_data + test_data
    assert len(all_data) == len(train_data) + len(valid_data) + len(test_data)

    #### 1. GET ALL UNIQUE ITEMS AND REMOVE GAPS FROM ITEMS
    # Union all items from sequence
    unique_items = set()
    for data_point in all_data:
        unique_items |= set(data_point[0])
    # Union all GT items
    unique_items = unique_items.union(data_point[1] for data_point in all_data)

    # Remove gaps from items
    item_to_lstm_idx = {item: idx for (idx, item) in enumerate(unique_items)}
    # NOTE: Crucially, we decrement by 1 in the reverse dict, and exclude the 0 padding item
    lstm_idx_to_df_item = {v: k - 1 for k, v in item_to_lstm_idx.items() if v != 0}

    #### 2. REINDEX THE ITEMS IN THE DATASET
    # Apply mapping on the data
    for data_point in all_data:
        sequence = data_point[0]
        gt = data_point[1]
        for i, item in enumerate(sequence):
            sequence[i] = item_to_lstm_idx[item]
        data_point[1] = item_to_lstm_idx[gt]

    ### 3. COMPUTE REINDEXED ITEMS TO COMMUNITY
    lstm_idx_to_community = {}
    df = original_df.groupby([item_key, community_key], as_index=False).size()
    df_item_to_community = dict(zip(df.streamer_name, df.community))
    for lstm_idx, df_item in lstm_idx_to_df_item.items():
        lstm_idx_to_community[lstm_idx] = df_item_to_community[df_item]

    ### 4. Checks on our dictionaries
    # Min of df is 0
    assert min(lstm_idx_to_df_item.values()) == 0
    # Min of lstm_idx mapping is 1
    assert min(lstm_idx_to_df_item.keys()) == 1

    return lstm_idx_to_community, unique_items, item_to_lstm_idx, lstm_idx_to_df_item


def load_community_dict(file_path):
    """Opens a pickled dictionary"""
    with open(file_path, "rb") as f:
        hm = pickle.load(f)
    return hm


def get_communities(sequence, community_dict):
    """Takes in sequence of items and returns list of communties

    The padding_idx should NOT have a mapping in the community_dict
    """
    return [community_dict[item] for item in sequence if item in community_dict]


def num_unique(communities, *args, **kwargs):
    """Returns the number of unique communities"""
    return len(set(communities))

# @cache
def shannon_index(communities, community_dict):
    """This is a metric of intra-list diversity https://en.wikipedia.org/wiki/Diversity_index#Shannon_index"""
    richness = len(set(community_dict.values()))
    hm_communities = Counter(communities)
    ans = 0
    for label in range(richness):
        if label in hm_communities:
            p = hm_communities[i] / len(communities)
            ans += p * math.log(p)
        else:
            ans += 0
    return -ans

def _simpson_index(communities, community_dict):
    """Gives more weight to dominant topics
    
    The original Simpson index λ equals the probability that two entities taken 
    at random from the dataset of interest (with replacement) represent the same type."""
    richness = len(set(community_dict.values()))
    hm_communities = Counter(communities)
    ans = 0
    for label in range(richness):
        if label in hm_communities:
            p = hm_communities[label] / len(communities)
            ans += p*p
        else:
            ans += 0
    return ans

# @cache        
def gini_simpson_index(communities, community_dict):
    return 1 - _simpson_index(communities, community_dict)
