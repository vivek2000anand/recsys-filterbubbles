from collections import defaultdict
import numpy as np


def printl(length=80):
    """Prents a horizontal line for prettier debugging"""
    print(length * "-")


def train_valid_test_split(
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
    # Only keep users with >= threshold items
    users, counts = np.unique(data[:, 0], return_counts=True)
    total_items_per_user = {
        user: count for user, count in zip(users, counts) if count >= min_items_per_user
    }
    count_per_user = defaultdict(int)

    new_data = []
    for user, item, time in data:
        if user not in total_items_per_user:
            continue
        else:
            total_items = total_items_per_user[user]

        count_per_user[user] += 1

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

    """\
    Description:
    ------------
        Input data for LSTM: Convert to user trajectory (maximum length: look back)
    """

    train, valid, test = [], [], []
    unique_users = set(data[:, 0])
    items_per_user = {int(user): [0 for i in range(look_back)] for user in unique_users}

    for (idx, row) in enumerate(data):
        user, item, time = int(row[0]), int(row[1]), row[2]
        # The item id increase happens here
        items_per_user[user] = items_per_user[user][1:] + [item + 1]
        current_items = items_per_user[user]
        if row[3] == 0:
            train.append([current_items[:-1], current_items[-1]])
        elif row[3] == 1:
            valid.append([current_items[:-1], current_items[-1]])
        else:
            test.append([current_items[:-1], current_items[-1]])

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
