from collections import defaultdict
import numpy as np


def printl(length=80):
    """Prents a horizontal line for prettier debugging"""
    print(length * "-")


def train_valid_test_split(data):
    """Takes in NX3 array and returns sequence-like data
    
    The data argument should be formatted as:
        Col 1: User ID
        Col 2: Item ID
        Col 3: Time
    """
    # Filter out users with < 10
    (users, counts) = np.unique(data[:, 0], return_counts=True)
    users = users[counts >= 10]

    # sequence_dictt = {int(user): [] for user in set(data[:, 0])}
    sequence_dict = defaultdict(list)

    # sequence_dict, pert_dic = {int(user): [] for user in set(data[:, 0])}, {
    #     int(user): [] for user in set(data[:, 0])
    # }

    
    user_dic = {int(user): idx for (idx, user) in enumerate(users)}

    new_data = []
    for i in range(data.shape[0]):
        user, item, time = data[i]
        # if int(data[i, 0]) in user_dic:
        #     new_data.append([int(data[i, 0]), int(data[i, 1]), data[i, 2], 0])

        if user in user_dic:
            new_data.append([user, item, time, 0])

    new_data = np.array(new_data)

    for i in range(new_data.shape[0]):
        user, item, time = new_data[i]
        sequence_dict[user].append([i, item, time])

    for user in sequence_dict.keys():
        cur_test = int(0.05 * len(sequence_dict[user]))
        for i in range(cur_test):
            interaction = sequence_dict[user].pop()
            new_data[interaction[0], 3] = 2

        # cur_val = int(0.1*len(sequence_dict[user]))
        # for i in range(cur_val):
        #    interaction = sequence_dict[user].pop()
        #    new_data[interaction[0],3] = 1

    return new_data


# NOTE: Sequence generator already increases item ids by 1!
def sequence_generator(data, look_back=50):

    """\
    Description:
    ------------
        Input data for LSTM: Convert to user trajectory (maximum length: look back)
    """

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
