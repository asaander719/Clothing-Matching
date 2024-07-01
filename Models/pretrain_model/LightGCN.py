import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

class LightGCN(nn.Module):
    def __init__(self, conf, pretrain_layer_num, user_embedding, item_embedding, device):
        super(LightGCN, self).__init__()
        self.train_data_path = conf["root_path"] + "/data/iqon_s/train.csv"
        self.device = device
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.num_layers = pretrain_layer_num
        self.adj_UJ, self.adj_IJ, self.all_top_ids, self.all_bottom_ids, self.all_users_ids, self.top_idx_to_encoded, self.bottom_idx_to_encoded = self.build_adj() 
        self.adj_IJ = self.adj_IJ.to(self.device)
        self.adj_UJ = self.adj_UJ.to(self.device)
        self.all_top_ids = self.all_top_ids.to(self.device)
        self.all_bottom_ids = self.all_bottom_ids.to(self.device)
        self.all_users_ids = self.all_users_ids.to(self.device)

    def build_adj(self):
        '''
        如果分别构建user-bottom的adj矩阵会出现u-b pair出现多次的情况则累加, t-b同理,ID不是从0开始的连续整数需要重新映射这些ID
        '''
        train_df = pd.read_csv(self.train_data_path, header=None).astype('int')
        train_df.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
        num_users = train_df['user_idx'].nunique()
        num_tops = train_df['top_idx'].nunique()
        num_bottoms = train_df['pos_bottom_idx'].nunique()

        all_top_ids = torch.LongTensor(train_df['top_idx'].unique())#.tolist()
        # all_bottom_ids = pd.concat([train_df['pos_bottom_idx'], train_df['neg_bottom_idx']]).unique()
        # only use positive bottoms to build adj, so we only update pos_bottoms instead of all bottoms to update item embs.
        all_bottom_ids = torch.LongTensor(train_df['pos_bottom_idx'].unique())#.tolist()
        all_users_ids = torch.LongTensor(train_df['user_idx'].unique())#.tolist()

        train_df['user_id_encoded'], _ = pd.factorize(train_df['user_idx'])
        train_df['top_id_encoded'], _ = pd.factorize(train_df['top_idx'])
        train_df['pos_bottom_id_encoded'], _ = pd.factorize(train_df['pos_bottom_idx'])

        ub_group = train_df.groupby(['user_id_encoded', 'pos_bottom_id_encoded']).size().reset_index(name='counts')
        tb_group = train_df.groupby(['top_id_encoded', 'pos_bottom_id_encoded']).size().reset_index(name='counts')
        # adj_UJ = csr_matrix((np.ones(len(train_df)), (train_df['user_idx'].values, train_df['pos_bottom_idx'].values)),
        #                                     shape=(num_users, num_bottoms))
        # adj_IJ = csr_matrix((np.ones(len(train_df)), (train_df['top_idx'].values, train_df['pos_bottom_idx'].values)),
        #                                     shape=(num_tops, num_bottoms))
        
        adj_UJ = csr_matrix((ub_group['counts'].values, (ub_group['user_id_encoded'].values, ub_group['pos_bottom_id_encoded'].values)),
            shape=(num_users, num_bottoms))
        adj_IJ = csr_matrix((tb_group['counts'].values, (tb_group['top_id_encoded'].values, tb_group['pos_bottom_id_encoded'].values)),
            shape=(num_tops, num_bottoms))

        adj_UJ = self.to_torch_sparse_tensor(adj_UJ)
        adj_IJ = self.to_torch_sparse_tensor(adj_IJ)
        top_idx_to_encoded = dict(zip(train_df['top_idx'], train_df['top_id_encoded']))
        bottom_idx_to_encoded = dict(zip(train_df['pos_bottom_idx'], train_df['pos_bottom_id_encoded']))
        return adj_UJ, adj_IJ, all_top_ids, all_bottom_ids, all_users_ids, top_idx_to_encoded, bottom_idx_to_encoded

    def to_torch_sparse_tensor(self, csr_matrix):
        #"将scipy.sparse的CSR矩阵转换为PyTorch稀疏张量"
        coo_matrix = csr_matrix.tocoo().astype(np.float32)
        row = torch.LongTensor(coo_matrix.row)
        col = torch.LongTensor(coo_matrix.col)
        index = torch.stack([row, col])
        value = torch.FloatTensor(coo_matrix.data)
        return torch.sparse.FloatTensor(index, value, torch.Size(coo_matrix.shape))
        
    def forward(self):
        top_embs = self.item_embedding(self.all_top_ids) 
        pos_bottoms_embs = self.item_embedding(self.all_bottom_ids)
        all_users_embs = self.user_embedding(self.all_users_ids)

        for _ in range(self.num_layers):
            user_emb_temp = torch.sparse.mm(self.adj_UJ, pos_bottoms_embs)  #user_num, hidden-dim
            pos_bottoms_emb_temp = torch.sparse.mm(self.adj_UJ.t(), all_users_embs)
            top_emb_temp = torch.sparse.mm(self.adj_IJ, pos_bottoms_embs)
            pos_bottoms_emb_temp += torch.sparse.mm(self.adj_IJ.t(), top_embs)

            # final_user_emb = user_emb.to(self.device) + user_emb_temp[torch.LongTensor(Us)]
            # final_Is_emb = Is_emb.to(self.device) + top_emb_temp[torch.LongTensor(Is)]
            # final_Js_emb = Js_emb.to(self.device) + pos_bottoms_emb_temp[torch.LongTensor(Js)]

        return user_emb_temp, top_emb_temp, pos_bottoms_emb_temp, self.top_idx_to_encoded, self.bottom_idx_to_encoded, top_embs, pos_bottoms_embs, all_users_embs
