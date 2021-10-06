import numpy as np
import torch
import sys
from utils import *
import rbo

FLOAT_MIN = -sys.float_info.max

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate): # wried, why fusion X 2?

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(torch.nn.Module):
    # required homebrewed mha layer for Ti/SASRec experiments
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) *  (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights) # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs


class TiSASRec(torch.nn.Module): # similar to torch.nn.MultiheadAttention
    def __init__(self, user_num, item_num, time_num, args,device,iteration,total_num):
        super(TiSASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = device
        self.random_seed = iteration
        self.total_num = total_num

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.abs_pos_K_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)

        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate,
                                                            device)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            self.pos_sigmoid = torch.nn.Sigmoid()
            self.neg_sigmoid = torch.nn.Sigmoid()

    def seq2feats(self, user_ids, log_seqs, time_matrices):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.item_emb_dropout(seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = torch.LongTensor(time_matrices).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](seqs) # PyTorch mha requires time first fmt
            mha_outputs = self.attention_layers[i](Q, seqs,
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V,
                                            abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs): # for training
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, time_matrices, item_indices): # for inference
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

    def traintest(self,dataset,perturbed_users,original_probs,original_rank,final_metrics,test_len,device,args):

        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.random_seed)
 
        [user_train,  test, usernum, itemnum, timenum,user_map,item_map] = dataset
        num_batch = len(user_train) // args.batch_size

        perturbed_dic = {}
        for user in perturbed_users: 
            perturbed_dic[user] = 1

        user_list = sorted(list(user_train.keys()))
        maxlen = args.maxlen
        total_interaction = self.total_num
        num_batch = (len(user_train) // args.batch_size)+1

        relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)

        epoch_start_idx = 1
        
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        adam_optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=(0.9, 0.98))

        for epoch in range(epoch_start_idx, args.epochs + 1):
            
            total_loss = 0
           
            for step in range(num_batch):
                st_idx,ed_idx = step*args.batch_size, (step+1)*args.batch_size
                if ed_idx>len(user_train):
                    ed_idx = len(user_train)
                cur_len = ed_idx-st_idx
                u = np.zeros((cur_len,1),dtype=np.int32)
                seq = np.zeros((cur_len,args.maxlen),dtype=np.int32)
                pos = np.zeros((cur_len,args.maxlen),dtype=np.int32)
                neg = np.zeros((cur_len,args.maxlen),dtype=np.int32)
                index = np.zeros((cur_len,args.maxlen),dtype=np.int32)
                time_seq = np.zeros((cur_len,args.maxlen),dtype=np.int32)
                time_matrix = np.zeros((cur_len,args.maxlen,args.maxlen),dtype=np.int32)

                for i in range(st_idx,ed_idx):
                    cur_idx = st_idx-i
                    user = user_list[i]
                    seqt = np.zeros([maxlen], dtype=np.int32)
                    time_seqt = np.zeros([maxlen], dtype=np.int32)
                    post = np.zeros([maxlen], dtype=np.int32)
                    negt = np.zeros([maxlen], dtype=np.int32)
                    indext = np.zeros([maxlen], dtype=np.int32)
                    nxt = user_train[user][-1][0]

                    idx = maxlen - 1
                    ts = {x[0]:1 for x in user_train[user]}
                    item_list = [item for item in range(1,itemnum+1) if item not in ts]
                    for i in reversed(user_train[user][:-1]):
                        seqt[idx] = i[0]
                        time_seqt[idx] = i[1]
                        post[idx] = nxt
                        indext[idx] = i[2]
                        if nxt != 0: negt[idx] = i[3]
                        nxt = i[0]
                        idx -= 1
                        if idx == -1: break
                    
                    u[cur_idx] = user
                    seq[cur_idx] = seqt
                    time_seq[cur_idx] = time_seqt
                    pos[cur_idx] = post
                    neg[cur_idx] = negt
                    index[cur_idx] = indext
                    time_matrix[cur_idx] = relation_matrix[user]

                pos_logits, neg_logits = self(u, seq, time_matrix, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
                # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in self.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                for param in self.abs_pos_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                for param in self.abs_pos_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                for param in self.time_matrix_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                for param in self.time_matrix_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                total_loss += loss.item()
       
            if epoch%10==0:
                print("training loss in epoch {}: {}".format(epoch, total_loss)) # expected 0.4~0.6 after init few epochs

        self.eval()
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
                  
            for i in reversed(user_train[u]):
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
                    softmax = torch.nn.Softmax(dim=1)
                    predictions = softmax(self.predict(u_array, seq_array, matrix_array, item_indices).cpu().detach()).numpy()

                    for j in range(count):
                        target_idx = int(truth[j])
                        pred = predictions[j]
                        current_val = pred[target_idx-1]
                        new_prob = pred - current_val
                        rank =  np.count_nonzero(new_prob>0)+1
                        predicted_rank[total_count-count+j] = rank
                        truth_item[total_count-count+j] = target_idx
                                
                        MRR += 1/rank
                        HITS += (1 if rank<=10 else 0)
                        probs[total_count-count+j] = pred
                        test_users[total_count-count+j] = u
                    
                    print('.',end='')
                    sys.stdout.flush()
                    count=0

                seq = seq[1:] + [test[u][i][0]]
                time_seq = time_seq[1:] + [test[u][i][1]]
        
        MRR /= total_test_num
        HITS /= total_test_num
        print('Test MRR = {}\tTest HITS = {}\n'.format(MRR,HITS))

        if original_probs!=-1 and epoch == args.epochs:                
            total_test_num = len(truth_item)
            rank1,rank2 = np.argsort(original_probs,axis=1)[:,::-1],np.argsort(probs,axis=1)[:,::-1]
            for i in range(total_test_num):
                if test_users[i] not in perturbed_dic:
                    ground_truth = truth_item[i]-1                     
                    MRR_diff = abs((1/predicted_rank[i])-(1/original_rank[i]))
                    HITS_diff = abs((1 if predicted_rank[i]<=10 else 0)-(1 if original_rank[i]<=10 else 0))
                    RBO = rbo.RankingSimilarity(rank1[i,:], rank2[i,:]).rbo()
                    rank_diff = abs(predicted_rank[i]-original_rank[i])
                    prob_diff = abs(probs[i][ground_truth] - original_probs[i][ground_truth])
                    jaccard = np.intersect1d(rank1[i,:10],rank2[i,:10]).shape[0]/np.union1d(rank1[i,:10],rank2[i,:10]).shape[0]
                    final_metrics[0].append(MRR_diff)
                    final_metrics[1].append(HITS_diff)
                    final_metrics[2].append(RBO)
                    final_metrics[3].append(rank_diff)
                    final_metrics[4].append(prob_diff)
                    final_metrics[5].append(jaccard)


        return [probs,predicted_rank,final_metrics,[MRR,HITS]] 
