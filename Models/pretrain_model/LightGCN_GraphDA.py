import torch
# from dataloader import BasicDataset
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SGConv

from gtn_propagation import GeneralPropagation
import warnings
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import torch_geometric
from torch_sparse import SparseTensor


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    

class LightGCN(BasicModel):
    def __init__(self, 
                 conf,
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.conf = conf
        # self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.conf.recdim #self.config['latent_dim_rec']
        self.n_layers = self.conf.layer #self.config['lightGCN_n_layers']
        self.keep_prob = self.conf.keepprob #self.config['keep_prob']
        self.A_split = False #self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)

    def reset_all(self):
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)

    def reset_all_uuii(self):
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet, include_uuii=True)


    def reset_graph(self):
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.conf.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getUsersUsers(self, users):
        all_users, _ = self.computer()
        users_emb = all_users[users.long()]

        return self.f(torch.matmul(users_emb, all_users.t()))

    def getItemsItems(self, items):
        _, all_items = self.computer()
        items_emb = all_items[items.long()]
        
        return self.f(torch.matmul(items_emb, all_items.t()))

    def getItemsRating(self, items):
        all_users, all_items = self.computer()
        items_emb = all_items[items.long()]
        users_emb = all_users
        item_rating = self.f(torch.matmul(items_emb, users_emb.t()))
        return item_rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
       
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma # U*I inner pro. Matrix
