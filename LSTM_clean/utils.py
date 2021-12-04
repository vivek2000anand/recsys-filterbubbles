from collections import defaultdict

import numpy as np
from torch._C import Value


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
        if count_per_user[user] <= total_items - 2:
            new_data.append([user, item, time, 0])
        elif count_per_user[user] == total_items - 1:
            new_data.append([user, item, time, 1])
        elif count_per_user[user] == total_items:
            new_data.append([user, item, time, 2])
        else:
            raise ValueError(
                "There's a bug in the counting of total items or curr items!"
            )

        # # Original Version
        # if count_per_user[user] <= train_cutoff * total_items:
        #     new_data.append([user, item, time, 0])
        # elif count_per_user[user] <= valid_cutoff * total_items:
        #     new_data.append([user, item, time, 1])
        # elif count_per_user[user] <= total_items:
        #     new_data.append([user, item, time, 2])
        # else:
        #     raise ValueError(
        #         "There's a bug in the counting of total items or curr items!"
        #     )

    return np.array(new_data)


# NOTE: Sequence generator already increases item ids by 1!
def sequence_generator(data, look_back=50):

    """\
    Description:
    ------------
        Input data for LSTM: Convert to user trajectory (maximum length: look back)
    """

    train, valid, test = [], [], []
    unique_users = set(data[:, 0])
    items_per_user = {int(user): [0 for _ in range(look_back)] for user in unique_users}

    for user, item, time, split in data:
        # NOTE: Item ID increase happens here
        items_per_user[user] = items_per_user[user][1:] + [item + 1]
        current_items = items_per_user[user]
        if split == 0:
            train.append([current_items[:-1], current_items[-1]])
        elif split == 1:
            valid.append([current_items[:-1], current_items[-1]])
        elif split == 2:
            test.append([current_items[:-1], current_items[-1]])
        else:
            raise ValueError("Some of the data has not been split into train/valid/test!")

    return train, valid, test


def get_diversity(prev_item_communities, predicted_item_communities, bounds=0.1):
    """Generates diversity of the recommendations

    Args:
        prev_item_communities ([type]): List of communities of the immediate previous items
        predicted_item_communities ([type]): List of topk communities of current items
        bounds (float, optional): Percentage for diverse or moderate. Defaults to 0.1.

    Returns:
        [type]: [description]
    """
    assert bounds < 1 and bounds > 0
    diversity = []
    top_length = len(predicted_item_communities[-1])
    for prev_item, pred_items in zip(prev_item_communities, predicted_item_communities):
        sum = 0
        # We check if the previous item community is the same as those in the topk predicted
        for item in pred_items:
            if prev_item == item:
                sum += 1
        if sum >= (1 - bounds) * top_length:
            # Too many within the same community (filter bubble)
            diversity.append(-1)
        elif sum <= bounds * top_length:
            # Very diverse recommendations
            diversity.append(1)
        else:
            # Neither filter bubble or very diverse
            diversity.append(0)
        print("New Item:")
        print("Prev Items: ", prev_item)
        print("Predicted Items: ", pred_items)
    return diversity
