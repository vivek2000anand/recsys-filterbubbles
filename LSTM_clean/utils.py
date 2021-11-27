import numpy as np

def train_test_split(data=[]):
    
    """\
    Description:
    ------------
        Sort by each user's interation. Tag train & test & (valid) set
    """

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
    
    for user in sequence_dic.keys():
        cur_test = int(0.05*len(sequence_dic[user]))
        for i in range(cur_test):
            interaction = sequence_dic[user].pop()
            new_data[interaction[0],3] = 2

        #cur_val = int(0.1*len(sequence_dic[user]))
        #for i in range(cur_val):
        #    interaction = sequence_dic[user].pop()
        #    new_data[interaction[0],3] = 1

    return new_data

def sequence_generator(data, look_back = 50):

    """\
    Description:
    ------------
        Input data for LSTM: Convert to user trajectory (maximum length: look back)
    """

    train,test, valid = [],[],[]
    train_items, test_items, valid_items = [], [], []
    unique_users = set(data[:,0])
    items_per_user = {int(user):[0 for i in range(look_back)] for user in unique_users}
    
    for (idx,row) in enumerate(data):
      user,item,time = int(row[0]),int(row[1]),row[2]
      items_per_user[user] = items_per_user[user][1:]+[item+1]
      current_items = items_per_user[user]
      if row[3]==0:
        train.append([current_items[:-1],current_items[-1]])
        train_items.append(item)
      elif row[3]==2:
        test.append([current_items[:-1],current_items[-1]])
        test_items.append(item)
      else:
        valid.append([current_items[:-1],current_items[-1]])
        valid_items.append(item)
                                                                
    return train,test, train_items, test_items #,valid


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
                sum +=1
        if sum >= (1-bounds)*top_length:
            # Too many within the same community (filter bubble)
            diversity.append(-1)
        elif sum <= bounds * top_length:
            # Very diverse recommendations
            diversity.append(1)
        else:
            # Neither filter bubble or very diverse
            diversity.append(0)
    return diversity