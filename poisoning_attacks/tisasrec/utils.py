import sys
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def computeRePos(time_seq, time_span):
    
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i]-time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix

def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, result_queue, SEED):
    def sample(user):

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]
    
        idx = maxlen - 1
        ts = set(map(lambda x: x[0],user_train[user]))
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        time_matrix = relation_matrix[user]
        return (user, seq, time_seq, time_matrix, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, batch_size=64, maxlen=10,n_workers=1,seed=0):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      relation_matrix,
                                                      self.result_queue,
                                                      seed
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set: # float as map key?
        time_map[time] = int(round(float(time-time_min)))
    return time_map

def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])

    user_set = sorted(user_set)
    item_set = sorted(item_set)
    
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u+1
    for i, item in enumerate(item_set):
        item_map[item] = i+1
    
    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]],x[2],item_map[x[3]],x[4]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list)-1):
            if time_list[i+1]-time_list[i] != 0:
                time_diff.add(time_list[i+1]-time_list[i])
        if len(time_diff)==0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1),x[2],x[3],x[4]], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max), user_map,item_map

def data_partition(data):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = defaultdict(list)
    user_test = defaultdict(list)
    
    print('Preparing data...')
    time_set = set()
    for (idx,row) in enumerate(data):
        u, i, timestamp,test,neg_item = row[0],row[1],row[2],row[3],row[4]
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        test = int(test)
        time_set.add(timestamp)
        User[u].append([i, timestamp,idx,neg_item,test])
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum,user_map,item_map = cleanAndsort(User, time_map)

    for user in User:
        for interaction in User[user]:
            if interaction[4]==0:
                user_train[user].append(interaction[:4])
            else:
                user_test[user].append(interaction[:3])
    print('Preparing done...')
    return [user_train, user_test, usernum, itemnum, timenum,user_map,item_map]


def evaluate(model, dataset, args):
    
    [train, test, usernum, itemnum, timenum, user_map,item_map ] = dataset

    total_test_num = sum([len(test[user]) for user in test.keys()])
    predicted_rank = [0 for i in range(total_test_num)]
    probs = [0 for i in range(total_test_num)]
    test_users = [0 for i in range(total_test_num)]
    truth_item = [0 for i in range(total_test_num)]

    MRR,HITS = 0,0
    
    batch_size = 1024
    u_array = np.zeros((batch_size,1),dtype=np.int32)
    seq_array = np.zeros((batch_size,args.maxlen),dtype=np.int32)
    matrix_array = np.zeros((batch_size,args.maxlen,args.maxlen),dtype=np.int32)
    truth = np.zeros(batch_size,dtype=np.int32)
    count = 0
    total_count = 0

    for u in test.keys():

        seq = [0 for i in range(args.maxlen)]
        time_seq = [0 for i in range(args.maxlen)]
        idx = args.maxlen - 1
              
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx]=i[1]
            idx -= 1
            if idx == -1: break

        item_indices = list(range(1, itemnum+1))

        for i in range(len(test[u])):
            target_idx = test[u][i][0]
            time_matrix = computeRePos(np.array(time_seq), args.time_span)
            
            u_array[count,0] = u
            seq_array[count] = np.array(seq)
            matrix_array[count] = time_matrix
            truth[count] = target_idx
            count += 1
            total_count += 1
            if count==batch_size or total_count == total_test_num:
                predictions = model.predict(u_array, seq_array, matrix_array, item_indices).cpu().detach().numpy()
                
                for j in range(count):
                    target_idx = int(truth[j])
                    pred = predictions[j]
                    current_val = pred[target_idx-1]
                    new_prob = pred - current_val
                    rank =  np.count_nonzero(new_prob>0)+1
                    predicted_rank[total_count-count+j] = rank
                    truth_item[total_count-count+j] = target_idx
                            
                    MRR += 1/rank
                    HITS += (1 if rank<10 else 0)
                    probs[total_count-count+j] = pred
                    test_users[total_count-count+j] = u
                
                print('.',end='')
                sys.stdout.flush()
                count=0

            seq = seq[1:] + [test[u][i][0]]
            time_seq = time_seq[1:] + [test[u][i][1]]
    
    MRR /= total_test_num
    HITS /= total_test_num

    print('MRR = {}\tHITS = {}\n'.format(MRR,HITS))
    return [probs,predicted_rank,test_users,truth_item,[MRR,HITS]]

