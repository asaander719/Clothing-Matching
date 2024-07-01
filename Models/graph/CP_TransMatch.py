import torch
from torch.nn import *
import torch.nn as nn
from torch.optim import Adam
from torch.nn.init import uniform_, normal_
from torch.nn import functional as F
import pdb
from abc import abstractmethod
import numpy as np
from pretrain.LightGCN import LightGCN
import pandas as pd
from scipy.sparse import csr_matrix

def bpr_loss(pos_score, neg_score):
    loss = - F.logsigmoid(pos_score - neg_score).mean()
    return loss

class Aggregator(nn.Module):
    def __init__(self, emb_dim, self_included, agg_param):
        super(Aggregator, self).__init__()
        self.emb_dim = emb_dim
        self.act = F.relu
        self.self_included = self_included
        self.agg_param = agg_param

    def forward(self, self_vectors, neighbor_entity_vectors, neighbor_edge_vectors, masks):
        # self_vectors: [batch_size, -1, emb_dim]
        # neighbor_edge_vectors: [batch_size, -1, 2, n_neighbor, emb_dim]
        # masks: [batch_size, -1, 2, n_neighbor, 1]
        nei_nums = torch.sum(masks, dim=-2)
        nei_nums[nei_nums == 0] = 1 #it happens when neighbor number is set small
        neighbor_edge_vectors = torch.sum(neighbor_edge_vectors * masks, dim=-2)/nei_nums  # [batch_size, -1, 2, input_dim]

        outputs = self._call(self_vectors, neighbor_entity_vectors, neighbor_edge_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, emb_dim]
        # entity_vectors: [batch_size, -1, 2, emb_dim]
        pass    

class ConcatAggregator(Aggregator):
    def __init__(self, emb_dim, self_included, agg_param):
        super(ConcatAggregator, self).__init__(emb_dim, self_included, agg_param)
        multiplier = 3 if self_included else 2
        self.layer = nn.Linear(self.emb_dim * multiplier, self.emb_dim)
        self.layer_entity = nn.Linear(2 * self.emb_dim, self.emb_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, neighbor_entity_vectors, neighbor_edge_vectors):
        # self_vectors: [batch_size, -1, emb_dim]
        # neighbor_entity_vectors: [batch_size, -1, 2, emb_dim]
        # neighbor_edge_vectors: [batch_size, -1, 2, emb_dim] # neighbor edges have been aggregated
        bs = self_vectors.size(0)
#         neighbor_vectors = neighbor_entity_vectors + self.agg_param * neighbor_edge_vectors # this is what we add [bs, -1, 2, emb_size]
        neighbor_vectors = torch.cat([neighbor_entity_vectors, self.agg_param * neighbor_edge_vectors], dim=-1)
        neighbor_vectors = self.layer_entity(neighbor_vectors)
        if self.self_included:
            neighbor_vectors_view = neighbor_vectors.view([bs, -1, self.emb_dim * 2]) # [bs, -1, emb_dim * 2]
            self_vectors = self_vectors.view([bs, -1, self.emb_dim])  # [bs, -1, emb_dim]
            if len(self_vectors.size()) < len(neighbor_vectors_view.size()):
                self_vectors = self_vectors.unsqueeze(-2)
            self_vectors = torch.cat([self_vectors, self.agg_param * neighbor_vectors_view], dim=-1)  # [bs, -1, emb_dim * 3]
        else:
            self_vectors = neighbor_vectors    
        self_vectors = self.layer(self_vectors)  # [bs, -1, emb_dim]
#         self_vectors = self.act(self_vectors)
        return self_vectors, neighbor_vectors


class MeanAggregator(Aggregator):
    def __init__(self, emb_dim, self_included, agg_param):
        super(MeanAggregator, self).__init__(emb_dim, self_included, agg_param)
        self.layer = nn.Linear(self.emb_dim, self.emb_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, neighbor_entity_vectors, neighbor_edge_vectors):
        bs = self_vectors.size(0)
        neighbor_vectors = neighbor_entity_vectors + self.agg_param * neighbor_edge_vectors
        if self.self_included:
            self_vectors = self_vectors.view([bs, -1, self.emb_dim])  # [bs, -1, emb_dim]
            if len(self_vectors.size()) < len(neighbor_vectors.size())-1:
                self_vectors = self_vectors + self.agg_param * torch.mean(neighbor_vectors, dim=-2).view([bs, -1, self.emb_dim])
            else:
                self_vectors = self_vectors + self.agg_param * torch.mean(neighbor_vectors, dim=-2)
        else:
            self_vectors = torch.mean(neighbor_vectors, dim=-2)
        return self_vectors, neighbor_vectors 
    

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, user_emb, top_k_emb):
        # Query: 原始用户, Key/Value: Top-K相似用户
        q = self.query(user_emb).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        k = self.key(top_k_emb)                # [batch_size, k, embedding_dim]
        v = self.value(top_k_emb)              # [batch_size, k, embedding_dim]

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)

        weighted_sum = torch.matmul(attention, v).squeeze(1)
        return weighted_sum

class TransMatch(Module):
    def __init__(self, conf, neighbor_params=None, visual_features=None):      
        super(TransMatch, self).__init__()
        self.hidden_dim = conf["hidden_dim"]
        self.user_num = conf["user_num"]
        self.item_num = conf["item_num"]
        self.batch_size = conf["batch_size"]
        self.device = conf["device"]
        self.score_type = conf["score_type"]
                 
        self.use_context = conf["context"]
        if self.use_context:
            self.context_hops = conf["context_hops"] 
        self.use_path = conf["path"]
        if self.use_path:
            self.path_weight = conf["path_weight"]
            self.path_agg = conf["path_agg"]
        self.use_selfatt = conf['use_selfatt']

        self.neighbor_params = neighbor_params
        self.entity2edges = torch.LongTensor(self.neighbor_params[0]).to(self.device)
        self.edge2entities = torch.LongTensor(self.neighbor_params[1]).to(self.device)
        self.edge2relation = torch.LongTensor(self.neighbor_params[2]).to(self.device)
        self.relation2entities = torch.LongTensor(self.neighbor_params[3]).to(self.device)
        self.entity2edge_set = self.neighbor_params[4]
        self.relation2entity_set = self.neighbor_params[5]
        
        self.userEmb = F.normalize(torch.normal(mean=torch.zeros(self.user_num + 1, self.hidden_dim), std=1/(self.hidden_dim)**0.5), p=2, dim=-1)
        self.itemEmb = F.normalize(torch.normal(mean=torch.zeros(self.item_num + 1, self.hidden_dim), std=1/(self.hidden_dim)**0.5), p=2, dim=-1)
        self.itemB = torch.zeros([self.item_num + 1, 1]).to(self.device)
        self.u_embeddings_l = nn.Embedding.from_pretrained(self.userEmb, freeze=False, padding_idx=self.user_num)
        self.i_bias_l = nn.Embedding.from_pretrained(self.itemB, freeze=False, padding_idx=self.item_num)
        self.i_embeddings_i = nn.Embedding.from_pretrained(self.itemEmb, freeze=False, padding_idx=self.item_num)

        self.pretrain_mode = conf["pretrain_mode"]
        self.pretrain_layer_num = conf['pretrain_layer_num']
        if self.pretrain_mode:
            self.top_k_u = conf["top_k_u"]
            self.top_k_i = conf["top_k_i"]
            self.lightgcn = LightGCN(conf, self.pretrain_layer_num, self.u_embeddings_l, self.i_embeddings_i, self.device)
            if conf['use_selfatt']:
                self.self_attention = SelfAttention(self.hidden_dim)
     
        self.visual_nn_comp = Sequential(
            Linear(conf["visual_feature_dim"], self.hidden_dim),
            nn.Sigmoid())

        self.visual_nn_comp[0].apply(lambda module: normal_(module.weight.data,mean=0,std=1/(self.hidden_dim)**0.5))
        self.visual_nn_comp[0].apply(lambda module: normal_(module.bias.data,mean=0,std=1/(self.hidden_dim)**0.5))

        self.visual_nn_per = Sequential(
            Linear(conf["visual_feature_dim"], self.hidden_dim),
            nn.Sigmoid())
        self.visual_nn_per[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.visual_nn_per[0].apply(lambda module: uniform_(module.bias.data,0,0.001))
        self.visual_features = visual_features
        self.i_bias_v = nn.Embedding.from_pretrained(self.itemB, freeze=False, padding_idx=self.item_num)
        self.u_embeddings_v = nn.Embedding.from_pretrained(self.userEmb, freeze=False, padding_idx=self.user_num)
        self._zeros = torch.zeros(self.hidden_dim).to(self.device)

        # self.margin = nn.Parameter(torch.tensor(1e-10)) 
        # self.margin = 1e-10

        if self.use_context:
            # define aggregators for each layer
            self.agg_param = conf["agg_param"]
            self.neighbor_samples = conf["neighbor_samples"]
            if conf["neighbor_agg"] == "mean":
                self.neighbor_agg = MeanAggregator
            elif conf["neighbor_agg"] == "concat":
                self.neighbor_agg = ConcatAggregator

            self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())
            if self.score_type == "mlp":
                self.scorer = nn.Linear(self.hidden_dim, 1)    
            
        else:
            if self.score_type == "mlp":
                self.layer = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
                self.scorer = nn.Linear(self.hidden_dim, 1)  

     
    def _get_neighbor_aggregators(self):
        aggregators = []  # store all aggregators
        for i in range(self.context_hops):
            aggregators.append(self.neighbor_agg(emb_dim=self.hidden_dim, self_included=True, agg_param=self.agg_param))
        return aggregators   

    def _get_entity_neighbors_and_masks(self, relations, entity_pairs, train_edges=None):
        bs = relations.size(0)
        edges_list = [relations]
        nodes_list = []
        masks = []
        if train_edges != None:
            train_edges = torch.unsqueeze(train_edges, -1)  # [batch_size, 1]
        if len(entity_pairs.size()) == 2: #training case
            mode = "train"
            self_entities = entity_pairs.unsqueeze(-1).expand(-1, -1, self.neighbor_samples).reshape(bs, -1)
        elif len(entity_pairs.size()) == 3: #inference case
            mode = "test"
            candi_num = entity_pairs.size(1)
            self_entities = entity_pairs.unsqueeze(-1).expand(-1, -1, -1, self.neighbor_samples).reshape(bs, candi_num, -1)
        for i in range(self.context_hops):
            if i == 0:
                neighbor_entities = entity_pairs # [bs, 2] in training, [bs, 100, 2] in inference
                nodes_list.append(neighbor_entities)
                if mode == "train":
                    neighbor_edges = torch.index_select(self.entity2edges, 0, neighbor_entities.view(-1)).view([bs, -1])
                elif mode == "test":
                    neighbor_edges = torch.index_select(self.entity2edges, 0, neighbor_entities.view(-1)).view([bs, candi_num, -1])
            else:
                if mode == "train":
                    neighbor_entities = torch.index_select(self.edge2entities, 0, edges_list[-1].view(-1)).reshape([bs, -1, 2]) #bs, -1, 2
                    nodes_list.append(neighbor_entities)
                    neighbor_edges = torch.index_select(self.entity2edges, 0, neighbor_entities.view(-1)).view([bs, -1])
               
                elif mode == "test":   
                    neighbor_entities = torch.index_select(self.edge2entities, 0, edges_list[-1].view(-1)).reshape([bs, candi_num, -1, 2])
                    nodes_list.append(neighbor_entities)
                    neighbor_edges = torch.index_select(self.entity2edges, 0, neighbor_entities.view(-1)).view([bs, candi_num, -1])
               
      
            edges_list.append(neighbor_edges)
            if train_edges != None: 
                mask = neighbor_edges - train_edges  # [batch_size, -1]
            else:
                mask = neighbor_edges
            mask = (mask != 0).float()
            masks.append(mask)
        # edge_list: [[bs,], [bs, 10], [bs, 100], ...]
        
        return edges_list, nodes_list, masks

     #ori user_emb using padding 

    def find_topk_similar_users(self, batch_user_embeddings, all_user_embeddings, k):
        similarity = torch.matmul(batch_user_embeddings, all_user_embeddings.t())
        # 取Top-K，不包括自己
        topk_values, topk_indices = torch.topk(similarity, k=k+1, largest=True, sorted=True)
        self_indices = torch.arange(batch_user_embeddings.size(0)).unsqueeze(1)
        self_indices = self_indices.to(self.device)
        topk_indices = topk_indices.to(self.device)
        # print(self_indices.device, topk_indices.device)
        topk_indices = topk_indices.where(topk_indices != self_indices, topk_indices[:, -1].unsqueeze(1))
        return topk_indices[:, 1:].to(self.device)

    def _aggregate_neighbors_train(self, edge_list, entity_list, mask_list, relation_features, entity_features, visual=False):
        bs = edge_list[0].size(0)
        if visual:
            entity_vectors = [self.visual_nn_comp(entity_features[entity_list[0]])]
        else:
            entity_vectors = [entity_features(entity_list[0])]
        
        edge_emb = relation_features(edge_list[0])
        # T = self.T.expand_as(edge_emb)  
        # edge_emb = edge_emb + T 
        edge_vectors = [edge_emb] # bs, candi_num, emb_dim

        for edges in edge_list[1:]: # len(edge_list) = self.context_hops+1, len(entity_list) = self.context_hops
            relations = torch.index_select(self.edge2relation, 0, edges.view(-1)).view(list(edges.shape)) #         
            edge_vectors.append(relation_features(relations)) # edge represent the implict interaction of user and entity, not a real vector
            
        for entities in entity_list[1:]:
            if visual:
                entity_vectors.append(self.visual_nn_comp(entity_features[entities]))
            else:
                entity_vectors.append(entity_features(entities))
        # shape of edge vectors:
        # [[batch_size, relation_dim],
        #  [batch_size, 2 * neighbor_samples, relation_dim],
        #  [batch_size, (2 * neighbor_samples) ^ 2, relation_dim],
        #  ...]
        
        for i in range(self.context_hops):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter = []
            node_vectors_next_iter = []
            neighbor_edge_shape = [bs, -1, 2, self.neighbor_samples, aggregator.emb_dim]
            neighbor_entity_shape = [bs, -1, 2, aggregator.emb_dim]
            masks_shape = [bs, -1, 2, self.neighbor_samples, 1]
            for hop in range(self.context_hops - i): # aggregate in inverse order
                self_edge_vectors, neighbor_entity_vectors = aggregator(self_vectors=edge_vectors[hop], neighbor_entity_vectors=entity_vectors[hop].view(neighbor_entity_shape), neighbor_edge_vectors=edge_vectors[hop + 1].view(neighbor_edge_shape), masks=mask_list[hop].view(masks_shape))
                                
                edge_vectors_next_iter.append(self_edge_vectors)
                node_vectors_next_iter.append(neighbor_entity_vectors)
            
            edge_vectors = edge_vectors_next_iter
            entity_vectors = node_vectors_next_iter
        # edge_vectos[0]: [self.batch_size, 1, self.n_relations]
#         res = edge_vectors[0].view([bs, self.n_relations])
        return edge_vectors[0], entity_vectors[0].squeeze(1)


    def _aggregate_neighbors_test(self, edge_list, entity_list, mask_list, relation_features, entity_features, visual=False):
        bs, candi_num = edge_list[0].size()
        if visual:
            entity_vectors = [self.visual_nn_comp(entity_features[entity_list[0]])]
        else:
            entity_vectors = [entity_features(entity_list[0])]

        edge_emb = relation_features(edge_list[0])
        # T = self.T.expand_as(edge_emb)  
        # edge_emb = edge_emb + T 
        edge_vectors = [edge_emb] 
        
        for edges in edge_list[1:]: # len(edge_list) = self.context_hops+1, len(entity_list) = self.context_hops
            relations = torch.index_select(self.edge2relation, 0, edges.view(-1)).view(list(edges.shape)) #    
            edge_vectors.append(relation_features(relations)) # initial state: edge vectors are pure relation features
        for entities in entity_list[1:]:
            if visual:
                entity_vectors.append(self.visual_nn_comp(entity_features[entities]))
            else:
                entity_vectors.append(entity_features(entities))

        for i in range(self.context_hops):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter, node_vectors_next_iter = [], []
            neighbor_edge_shape = [bs, candi_num, -1, 2, self.neighbor_samples, aggregator.emb_dim]
            neighbor_entity_shape = [bs, candi_num, -1, 2, aggregator.emb_dim]
            masks_shape = [bs, candi_num, -1, 2, self.neighbor_samples, 1]
            for hop in range(self.context_hops - i): # aggregate in inverse order
                self_edge_vectors, neighbor_entity_vectors = aggregator(self_vectors=edge_vectors[hop], neighbor_entity_vectors=entity_vectors[hop].view(neighbor_entity_shape), neighbor_edge_vectors=edge_vectors[hop + 1].view(neighbor_edge_shape), masks=mask_list[hop].view(masks_shape))
                edge_vectors_next_iter.append(self_edge_vectors)
                node_vectors_next_iter.append(neighbor_entity_vectors)
            
            edge_vectors = edge_vectors_next_iter
            entity_vectors = node_vectors_next_iter
        return edge_vectors[0], entity_vectors[0].squeeze(-3)
    
    def aggregate_embeddings(self, user_embeddings, topk_indices, batch_user_embeddings):
        topk_user_embeddings = user_embeddings[topk_indices]
        aggregated_embeddings = self.self_attention(batch_user_embeddings, topk_user_embeddings)
        return aggregated_embeddings
    
    def aggregate_item_embeddings(self, item_embeddings, topk_indices, batch_item_embeddings):
        topk_item_embeddings = item_embeddings(topk_indices)
        aggregated_embeddings = self.self_attention(batch_item_embeddings, topk_item_embeddings)
        return aggregated_embeddings

    def transE_predict(self, u_rep, i_rep, j_rep, j_bias):
        '''
            u_rep: The embedding representation of a user,
            i_rep: The embedding representation of positive bottom.
            j_rep: The embedding representation of a negative bottom.
            j_bias: A bias term associated with the negative sample j_rep. 
                    Subtracted from sum, to adjust the score based on some intrinsic property of the negative sample.
        '''
        pred = j_bias - torch.sum(torch.pow((u_rep + i_rep - j_rep), 2), -1, keepdim=True)
        # computes the element-wise square of the difference, output a single score for each triple.
        # experiment shows that j_bias is quite effective, and ||Mrh+r-Mrt|| is effective than h*r*t (DistMult).
        return pred.squeeze(-1)
    
    
    def get_path_rep(self, paths, path_mask, rel_rep):
        # paths: bs, path_num, path_len
        path_emb = self.u_embeddings_l(paths)
        # T = self.T.expand_as(path_emb)  
        # path_emb = path_emb + T 

        if self.path_agg == "mean":

            path_rep = torch.mean(path_emb * path_mask.unsqueeze(-1), (-2, -3)) 
        elif self.path_agg == "sum":
            path_rep = torch.sum(path_emb * path_mask.unsqueeze(-1), (-2, -3)) / torch.clamp(torch.sum(path_mask, (-1, -2)), min=1).unsqueeze(-1)
        elif self.path_agg == "att":
            # rel_rep: bs, 1， emb_size (train); bs, 2, emb_size (inference)
            path_rep = torch.sum(path_emb * path_mask.unsqueeze(-1), dim=-2) / torch.clamp(torch.sum(path_mask, dim=-1), min=1).unsqueeze(-1) # bs, path_num, em_dim
            path_num = paths.size(-2)
            if len(path_rep.size()) == 3:
                rel_path = torch.exp(torch.matmul(rel_rep, path_rep.permute(0,2,1))).squeeze(-2) #bs, path_num
                coef = rel_path / torch.sum(rel_path, dim=-1).unsqueeze(-1).expand(-1, path_num) #bs, path_num
            elif len(path_rep.size()) == 4:
                rel_path = torch.exp(torch.matmul(rel_rep.unsqueeze(-2), path_rep.permute(0,1,3,2))).squeeze(-2) #bs, path_num
                coef = rel_path / torch.sum(rel_path, dim=-1).unsqueeze(-1).expand(-1, -1, path_num)
            path_rep = torch.sum(coef.unsqueeze(-1) * path_rep, dim=-2)
        return path_rep
    
    def forward(self, batch):
        Us = batch[0]
        Is = batch[1]
        Js = batch[2] 
        Ks = batch[3]

        if self.use_context:
            self.entity_pairs_pos = torch.cat([Is.unsqueeze(1), Js.unsqueeze(1)], dim=-1)
            self.entity_pairs_neg = torch.cat([Is.unsqueeze(1), Ks.unsqueeze(1)], dim=-1) # bs, 2
            self.train_edges = batch[4]       
            edge_list_pos, entity_list_pos, mask_list_pos = self._get_entity_neighbors_and_masks(Us, self.entity_pairs_pos, self.train_edges)     
            edge_pos_rep, entity_pos_rep = self._aggregate_neighbors_train(edge_list_pos, entity_list_pos, mask_list_pos, self.u_embeddings_l, self.i_embeddings_i)
            # edge_rep: bs, emb_dim; entity_rep: bs, 2, emb_dim
            # edge_rep can be directly used to predict the score
            edge_list_neg, entity_list_neg, mask_list_neg = self._get_entity_neighbors_and_masks(Us, self.entity_pairs_neg)
            edge_neg_rep, entity_neg_rep = self._aggregate_neighbors_train(edge_list_neg, entity_list_neg, mask_list_neg, self.u_embeddings_l, self.i_embeddings_i)
            U_latent_pos = edge_pos_rep
            U_latent_neg = edge_neg_rep
            I_latent_pos = entity_pos_rep[:,0,:]
            J_latent = entity_pos_rep[:,1,:]
            I_latent_neg = entity_neg_rep[:,0,:]
            K_latent = entity_neg_rep[:,1,:]
            
            if self.score_type == "mlp":
                R_j = self.scorer(U_latent_pos).squeeze(-1)
                R_k = self.scorer(U_latent_neg).squeeze(-1) #or apply classification loss
            elif self.score_type == "transE":
                J_bias_l = self.i_bias_l(Js)
                K_bias_l = self.i_bias_l(Ks)
                R_j = self.transE_predict(U_latent_pos.squeeze(-2), I_latent_pos, J_latent, J_bias_l)
                R_k = self.transE_predict(U_latent_neg.squeeze(-2), I_latent_neg, K_latent, K_bias_l)  
            
        else:
            U_latent = self.u_embeddings_l(Us)
            # T = self.T.expand_as(U_latent)  # [B H]
            # U_latent = U_latent + T 
            
            I_latent = self.i_embeddings_i(Is)
            J_latent = self.i_embeddings_i(Js)
            K_latent = self.i_embeddings_i(Ks)
            
            if self.score_type == "mlp":
                edge_pos_rep = torch.cat([U_latent, I_latent, J_latent], dim=-1)
                edge_neg_rep = torch.cat([U_latent, I_latent, K_latent], dim=-1)
                edge_pos_rep = self.layer(edge_pos_rep)  # [bs, -1, emb_dim]
                edge_pos_rep = F.relu(edge_pos_rep)
                edge_neg_rep = self.layer(edge_neg_rep)  # [bs, -1, emb_dim]
                edge_neg_rep = F.relu(edge_neg_rep)
                R_j = self.scorer(edge_pos_rep).squeeze(-1)
                R_k = self.scorer(edge_neg_rep).squeeze(-1)
                
            elif self.score_type == "transE":
                J_bias_l = self.i_bias_l(Js)
                K_bias_l = self.i_bias_l(Ks)
                R_j = self.transE_predict(U_latent, I_latent, J_latent, J_bias_l)
                R_k = self.transE_predict(U_latent, I_latent, K_latent, K_bias_l)
           
            U_latent_pos = U_latent
            U_latent_neg = U_latent
            I_latent_pos = I_latent
            I_latent_neg = I_latent       
                
        if self.pretrain_mode:
            user_emb_temp, top_emb_temp, pos_bottoms_emb_temp, top_idx_to_encoded, bottom_idx_to_encoded,top_embs, pos_bottoms_embs, all_users_embs = self.lightgcn.forward()

            U_latent = self.u_embeddings_l(Us)
            # T = self.T.expand_as(U_latent)  # [B H]
            # U_latent = U_latent + T 
            I_latent = self.i_embeddings_i(Is)
            J_latent = self.i_embeddings_i(Js)
            K_latent = self.i_embeddings_i(Ks).squeeze(1)
           
            U_latent = U_latent + user_emb_temp[Us]
            encode_Is= torch.LongTensor([top_idx_to_encoded[idx] for idx in Is.tolist()])#.to(self.device)
            I_latent = I_latent + top_emb_temp[encode_Is]
            encode_Js= torch.LongTensor([bottom_idx_to_encoded[idx] for idx in Js.tolist()])#.to(self.device)
            J_latent = J_latent + pos_bottoms_emb_temp[encode_Js]
          
            var_Ks_list = []
            for idx in Ks.tolist():
                if idx in bottom_idx_to_encoded:
                    var = pos_bottoms_emb_temp[torch.LongTensor([bottom_idx_to_encoded[idx]]).to(self.device)].squeeze(0)
                else:
                    var = self._zeros
                var_Ks_list.append(var)
            var_Ks = torch.stack(var_Ks_list, dim=0)
            K_latent = K_latent + var_Ks

            topk_users_idxs = self.find_topk_similar_users(U_latent, user_emb_temp, self.top_k_u)
            topk_Is_idxs = self.find_topk_similar_users(I_latent, top_emb_temp, self.top_k_i)
            topk_Js_idxs = self.find_topk_similar_users(J_latent, pos_bottoms_emb_temp, self.top_k_i)
            topk_Ks_idxs = self.find_topk_similar_users(K_latent, pos_bottoms_emb_temp, self.top_k_i)
          
            if self.use_selfatt:
                U_latent_pos  = self.aggregate_embeddings(user_emb_temp, topk_users_idxs, U_latent)  
                U_latent_neg = U_latent_pos    
                I_latent_pos = self.aggregate_embeddings(top_emb_temp, topk_Is_idxs, I_latent)
                I_latent_neg = I_latent_pos
                J_latent = self.aggregate_embeddings(pos_bottoms_emb_temp, topk_Js_idxs, J_latent)
                K_latent = self.aggregate_embeddings(pos_bottoms_emb_temp, topk_Ks_idxs, K_latent)

            if self.score_type == "mlp":
                R_j += self.scorer(U_latent_pos).squeeze(-1)
                R_k += self.scorer(U_latent_neg).squeeze(-1) #or apply classification loss
            elif self.score_type == "transE":
                J_bias_l = self.i_bias_l(Js)
                K_bias_l = self.i_bias_l(Ks)
                U_latent_pos = F.normalize(U_latent_pos.squeeze(-2),dim=0)
                U_latent_neg = F.normalize(U_latent_neg.squeeze(-2),dim=0)
                I_latent_pos = F.normalize(I_latent_pos,dim=0)
                I_latent_neg = F.normalize(I_latent_neg,dim=0)
                J_latent = F.normalize(J_latent,dim=0)
                K_latent = F.normalize(K_latent,dim=0)
                R_j += self.transE_predict(U_latent_pos, I_latent_pos, J_latent, J_bias_l)
                R_k += self.transE_predict(U_latent_neg, I_latent_neg, K_latent, K_bias_l)  
                        
        if self.use_path:
            pos_paths = batch[5]
            neg_paths = batch[6]
            pos_path_mask = batch[7]
            neg_path_mask = batch[8]

            pos_path_rep = self.get_path_rep(pos_paths, pos_path_mask, U_latent_pos) # bs, path_num, path_len
            neg_path_rep = self.get_path_rep(neg_paths, neg_path_mask, U_latent_neg) # bs, path_num, path_len
            
            R_j_p = self.transE_predict(pos_path_rep, I_latent_pos, J_latent, J_bias_l)
            R_k_p = self.transE_predict(neg_path_rep, I_latent_neg, K_latent, K_bias_l)

            R_j += R_j_p * self.path_weight
            R_k += R_k_p * self.path_weight
    
    
        J_bias_v = self.i_bias_v(Js)
        K_bias_v = self.i_bias_v(Ks)


        if self.use_context:
            edge_pos_rep_v, entity_pos_rep_v = self._aggregate_neighbors_train(edge_list_pos, entity_list_pos, mask_list_pos, self.u_embeddings_v, self.visual_features, True) 
            edge_neg_rep_v, entity_neg_rep_v = self._aggregate_neighbors_train(edge_list_neg, entity_list_neg, mask_list_neg, self.u_embeddings_v, self.visual_features, True)

            if self.score_type == "mlp":
                R_j = self.scorer(edge_pos_rep_v).squeeze(-1)
                R_k = self.scorer(edge_neg_rep_v).squeeze(-1) #or apply ification loss
            elif self.score_type == "transE":
                R_j_v = self.transE_predict(edge_pos_rep_v.squeeze(-2), entity_pos_rep_v[:,0,:], entity_pos_rep_v[:,1,:], J_bias_v)
                R_k_v = self.transE_predict(edge_neg_rep_v.squeeze(-2), entity_neg_rep_v[:,0,:], entity_neg_rep_v[:,1,:], K_bias_v)
        else:
            U_visual = self.u_embeddings_v(Us)
            vis_I = self.visual_features[Is]
            vis_J = self.visual_features[Js]
            vis_K = self.visual_features[Ks]
            I_visual = self.visual_nn_comp(vis_I) #bs, hidden_dim
            J_visual = self.visual_nn_comp(vis_J)
            K_visual = self.visual_nn_comp(vis_K)
            if self.score_type == "mlp":
                pass
            elif self.score_type == "transE":
                R_j_v = self.transE_predict(U_visual, I_visual, J_visual, J_bias_v)
                R_k_v = self.transE_predict(U_visual, I_visual, K_visual, K_bias_v)
            
        R_j += R_j_v
        R_k += R_k_v
        loss = bpr_loss(R_j, R_k)
        # loss = -torch.log(self.margin + torch.sigmoid(R_j - R_k)).mean()
        # 只有当R_j比R_k的score至少大出margin值时，loss才会更小,以防止negative sample和positive sample过于close
        # but only useful when those two are close, not effective if most of samples are not close

        return loss
    
    
    def cal_c_loss(self, pos, neg, anchor=None):
        # pos: [batch_size, pos_path_num, emb_size]
        # aug: [batch_size, neg_path_num, emb_size]
        if anchor is not None:
            pos_score = torch.matmul(anchor, pos.permute(0,2,1))
            neg_score = torch.matmul(anchor, neg.permute(0,2,1))
        else:
            pos_score = torch.matmul(pos, pos.permute(0, 2, 1)) # bs, pos_path_num, pos_path_num
            neg_score = torch.matmul(pos, neg.permute(0, 2, 1)) # bs, pos_path_num, neg_path_num
        pos_score = torch.sum(torch.exp(pos_score / self.c_temp), dim=-1) # bs, pos_num
        neg_score = torch.sum(torch.exp(neg_score / self.c_temp), dim=-1) # bs, pos_num
        ttl_score = pos_score + neg_score
        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def find_topk_similar_users_infer(self, batch_user_embeddings, all_user_embeddings, k, device):
        batch_size, _, emb_dim = batch_user_embeddings.shape
        reshaped_user_embeddings = batch_user_embeddings.view(-1, emb_dim)
        similarity = torch.matmul(reshaped_user_embeddings, all_user_embeddings.t())
        similarity = similarity.view(batch_size, _, -1) # 将相似度重塑回 [batch_size, 2, usernum]
        topk_values, topk_indices = torch.topk(similarity, k=k+1, largest=True, sorted=True)
        self_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(-1, 2, -1).to(device)
        topk_indices = topk_indices.where(topk_indices != self_indices, topk_indices[..., -1].unsqueeze(-1))
        return topk_indices[..., 1:] 
        
    def inference(self, batch):
        Us = batch[0]
        Is = batch[1]
        Js = batch[2] 
        Ks = batch[3] # torch.Size([1024, 1])

        J_list = torch.cat([Js.unsqueeze(1), Ks], dim=-1)
        j_num = J_list.size(1)
        J_bias_l = self.i_bias_l(J_list)
        if self.pretrain_mode:
            user_emb_temp, top_emb_temp, pos_bottoms_emb_temp, top_idx_to_encoded, bottom_idx_to_encoded,top_embs, pos_bottoms_embs, all_users_embs = self.lightgcn.forward()
        
            U_latent = self.u_embeddings_l(Us) #bs, j_num, hid
            # T = self.T.expand_as(U_latent)  # [B H]
            # U_latent = U_latent + T 
            I_latent = self.i_embeddings_i(Is)
            J_latent = self.i_embeddings_i(Js)
            K_latent = self.i_embeddings_i(Ks).squeeze(1)
            # Js_latent_list = self.i_embeddings_i(J_list)  #torch.Size([1024, 2, 32])

            U_latent = U_latent + user_emb_temp[Us]
            var_Is_list = []
            for idx in Is.tolist():
                if idx in top_idx_to_encoded:
                    var = top_emb_temp[torch.LongTensor([top_idx_to_encoded[idx]]).to(self.device)].squeeze(0)
                else:
                    var = self._zeros
                var_Is_list.append(var)
            var_Is = torch.stack(var_Is_list, dim=0)
            I_latent = I_latent + var_Is

            var_Js_list = []
            for idx in Js.tolist():
                if idx in bottom_idx_to_encoded:
                    var = pos_bottoms_emb_temp[torch.LongTensor([bottom_idx_to_encoded[idx]]).to(self.device)].squeeze(0)
                else:
                    var = self._zeros
                var_Js_list.append(var)
            var_Js = torch.stack(var_Js_list, dim=0)
            J_latent = J_latent + var_Js

            var_Ks_list = []
            for idx in Ks.squeeze(1).tolist():
                if idx in bottom_idx_to_encoded:
                    var = pos_bottoms_emb_temp[torch.LongTensor([bottom_idx_to_encoded[idx]]).to(self.device)].squeeze(0)
                else:
                    var = self._zeros
                var_Ks_list.append(var)
            var_Ks = torch.stack(var_Ks_list, dim=0)
            K_latent = K_latent + var_Ks
           
            topk_users_idxs = self.find_topk_similar_users(U_latent, user_emb_temp, self.top_k_u)
            topk_Is_idxs = self.find_topk_similar_users(I_latent, top_emb_temp, self.top_k_i)
            topk_Js_idxs = self.find_topk_similar_users(J_latent, pos_bottoms_emb_temp, self.top_k_i)
            topk_Ks_idxs = self.find_topk_similar_users(K_latent, pos_bottoms_emb_temp, self.top_k_i)
            if self.use_selfatt:
                U_latent  = self.aggregate_embeddings(user_emb_temp, topk_users_idxs, U_latent)      
                I_latent = self.aggregate_embeddings(top_emb_temp, topk_Is_idxs, I_latent)
                J_latent = self.aggregate_embeddings(pos_bottoms_emb_temp, topk_Js_idxs, J_latent)
                K_latent = self.aggregate_embeddings(pos_bottoms_emb_temp, topk_Ks_idxs, K_latent)
            
                U_latent = F.normalize(U_latent, dim=0)
                I_latent = F.normalize(I_latent, dim=0)
                J_latent = F.normalize(J_latent, dim=0)
                K_latent = F.normalize(K_latent, dim=0)

                U_latent = U_latent.unsqueeze(1).expand(-1, j_num, -1)
                I_latent = I_latent.unsqueeze(1).expand(-1, j_num, -1)

            Js_latent_ii = torch.stack((J_latent, K_latent), dim=1)
               
            if self.score_type == "mlp":
                edge_rep = torch.cat([U_latent, I_latent, Js_latent_ii], dim=-1)
                edge_rep = self.layer(edge_rep)  # [bs, -1, emb_dim]
                edge_rep = F.relu(edge_rep)
                scores += self.scorer(edge_rep).squeeze(-1)
            elif self.score_type == "transE": 
                J_bias_l = self.i_bias_l(J_list)
                # print(U_latent.size(), I_latent.size(), Js_latent_ii.size(), J_bias_l.size())
                scores = self.transE_predict(U_latent, I_latent, Js_latent_ii, J_bias_l)

        Us = Us.unsqueeze(1).expand(-1, j_num) #bs, j_num
        Is = Is.unsqueeze(1).expand(-1, j_num)
        
        if self.use_context:
            self.entity_pairs = torch.cat([Is.unsqueeze(-1), J_list.unsqueeze(-1)], dim=-1) # bs, j_num, 2
            edge_list, entity_list, mask_list = self._get_entity_neighbors_and_masks(Us, self.entity_pairs)
            edge_rep, entity_rep = self._aggregate_neighbors_test(edge_list, entity_list, mask_list, self.u_embeddings_l, self.i_embeddings_i)
            U_latent = edge_rep.squeeze(-2)
            I_latent_ii = entity_rep[:,:,0,:]
            Js_latent_ii = entity_rep[:,:,1,:]
        else:
            U_latent = self.u_embeddings_l(Us) 
            # T = self.T.expand_as(U_latent)  # [B H]
            # U_latent = U_latent + T 
            I_latent_ii = self.i_embeddings_i(Is)
            Js_latent_ii = self.i_embeddings_i(J_list) 
            if self.score_type == "mlp":
                edge_rep = torch.cat([U_latent, I_latent_ii, Js_latent_ii], dim=-1)
                edge_rep = self.layer(edge_rep)  # [bs, -1, emb_dim]
                edge_rep = F.relu(edge_rep)
 
        if self.score_type == "mlp":
            scores += self.scorer(edge_rep).squeeze(-1)
        elif self.score_type == "transE": 
            J_bias_l = self.i_bias_l(J_list)
            scores += self.transE_predict(U_latent, I_latent_ii, Js_latent_ii, J_bias_l) #torch.Size([1024, 2, 32])
            
        if self.use_path:
            j_paths = batch[4]
            k_paths = batch[5]
            j_path_mask = batch[6]
            k_path_mask = batch[7]
            
            if len(k_paths.size()) == 3:
                jk_paths = torch.cat([j_paths.unsqueeze(1), k_paths.unsqueeze(1)], dim=1) # [bs, 2, path_num, path_len]
                jk_path_mask = torch.cat([j_path_mask.unsqueeze(1), k_path_mask.unsqueeze(1)], dim=1) # 
                
            elif len(k_paths.size()) == 4:
                jk_paths = torch.cat([j_paths.unsqueeze(1), k_paths], dim=1)
                jk_path_mask = torch.cat([j_path_mask.unsqueeze(1), k_path_mask], dim=1) # 
                
            path_rep = self.get_path_rep(jk_paths, jk_path_mask, U_latent)
            self.p_scores = self.transE_predict(path_rep, I_latent_ii, Js_latent_ii, J_bias_l)
            scores += self.p_scores
 
        if self.use_context:
            edge_rep_v, entity_rep_v = self._aggregate_neighbors_test(edge_list, entity_list, mask_list, self.u_embeddings_v, self.visual_features, True)
            U_visual = edge_rep_v.squeeze(-2)
            I_visual_ii = entity_rep_v[:,:,0,:]
            Js_visual_ii = entity_rep_v[:,:,1,:] 
        else:
            U_visual = self.u_embeddings_v(Us)
            vis_I = self.visual_features[Is]
            vis_Js = self.visual_features[J_list]
            I_visual_ii = self.visual_nn_comp(vis_I) #bs, hidden_dim
            Js_visual_ii = self.visual_nn_comp(vis_Js)#bs, j_num, hidden_dim

        if self.score_type == "mlp":
            scores += self.scorer(edge_rep_v).squeeze(-1)
        elif self.score_type == "transE":
            J_bias_v = self.i_bias_v(J_list)
            self.vis_scores = self.transE_predict(U_visual, I_visual_ii, Js_visual_ii, J_bias_v)
            scores += self.vis_scores
        return scores
   