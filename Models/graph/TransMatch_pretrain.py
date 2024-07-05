import os
import pdb
from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn
from torch.nn import *
from torch.nn import functional as F
from torch.nn.init import normal_, uniform_
from torch.optim import Adam

from Models.graph.TransE import TransE
from Models.graph.TransR import TransR
from Models.graph.TransMatch_ori import TransMatch


def bpr_loss(pos_score, neg_score):
    loss = -F.logsigmoid(pos_score - neg_score)
    loss = torch.mean(loss)

    return loss


class Aggregator(nn.Module):

    def __init__(self, emb_dim, self_included, agg_param):
        super().__init__()
        self.emb_dim = emb_dim
        self.act = F.relu
        self.self_included = self_included
        self.agg_param = agg_param

    def forward(self, self_vectors, neighbor_entity_vectors,
                neighbor_edge_vectors, masks):
        # self_vectors: [batch_size, -1, emb_dim]
        # neighbor_edge_vectors: [batch_size, -1, 2, n_neighbor, emb_dim]
        # masks: [batch_size, -1, 2, n_neighbor, 1]
        nei_nums = torch.sum(masks, dim=-2)
        nei_nums[nei_nums ==
                 0] = 1  #it happens when neighbor number is set small
        neighbor_edge_vectors = torch.sum(
            neighbor_edge_vectors * masks,
            dim=-2) / nei_nums  # [batch_size, -1, 2, input_dim]

        outputs = self._call(self_vectors, neighbor_entity_vectors,
                             neighbor_edge_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, emb_dim]
        # entity_vectors: [batch_size, -1, 2, emb_dim]
        pass


class ConcatAggregator(Aggregator):

    def __init__(self, emb_dim, self_included, agg_param):
        super().__init__(emb_dim, self_included,
                                               agg_param)
        multiplier = 3 if self_included else 2
        self.layer = nn.Linear(self.emb_dim * multiplier, self.emb_dim)
        self.layer_entity = nn.Linear(2 * self.emb_dim, self.emb_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, neighbor_entity_vectors,
              neighbor_edge_vectors):
        # self_vectors: [batch_size, -1, emb_dim]
        # neighbor_entity_vectors: [batch_size, -1, 2, emb_dim]
        # neighbor_edge_vectors: [batch_size, -1, 2, emb_dim] # neighbor edges have been aggregated
        bs = self_vectors.size(0)
        #         neighbor_vectors = neighbor_entity_vectors + self.agg_param * neighbor_edge_vectors # this is what we add [bs, -1, 2, emb_size]
        neighbor_vectors = torch.cat(
            [neighbor_entity_vectors, self.agg_param * neighbor_edge_vectors],
            dim=-1)
        neighbor_vectors = self.layer_entity(neighbor_vectors)
        if self.self_included:
            neighbor_vectors_view = neighbor_vectors.view(
                [bs, -1, self.emb_dim * 2])  # [bs, -1, emb_dim * 2]
            self_vectors = self_vectors.view([bs, -1, self.emb_dim
                                              ])  # [bs, -1, emb_dim]
            if len(self_vectors.size()) < len(neighbor_vectors_view.size()):
                self_vectors = self_vectors.unsqueeze(-2)
            self_vectors = torch.cat(
                [self_vectors, self.agg_param * neighbor_vectors_view],
                dim=-1)  # [bs, -1, emb_dim * 3]
        else:
            self_vectors = neighbor_vectors
        self_vectors = self.layer(self_vectors)  # [bs, -1, emb_dim]
        #         self_vectors = self.act(self_vectors)
        return self_vectors, neighbor_vectors


class MeanAggregator(Aggregator):

    def __init__(self, emb_dim, self_included, agg_param):
        super().__init__(emb_dim, self_included, agg_param)
        # self.layer = nn.Linear(self.emb_dim, self.emb_dim)
        # nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, neighbor_entity_vectors,
              neighbor_edge_vectors):
        bs = self_vectors.size(0)
        neighbor_vectors = neighbor_entity_vectors + self.agg_param * neighbor_edge_vectors
        if self.self_included:
            self_vectors = self_vectors.view([bs, -1, self.emb_dim
                                              ])  # [bs, -1, emb_dim]
            if len(self_vectors.size()) < len(neighbor_vectors.size()) - 1:
                self_vectors = self_vectors + self.agg_param * torch.mean(
                    neighbor_vectors, dim=-2).view([bs, -1, self.emb_dim])
            else:
                self_vectors = self_vectors + self.agg_param * torch.mean(
                    neighbor_vectors, dim=-2)
        else:
            self_vectors = torch.mean(neighbor_vectors, dim=-2)
        return self_vectors, neighbor_vectors


class Attention(nn.Module):

    def __init__(self,
                 embedding_dim,
                 attn_dropout_prob=0.2,
                 hidden_dropout_prob=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        # self.attn_dropout = nn.Dropout(p=attn_dropout_prob)
        # # self.out_dropout = nn.Dropout(p=hidden_dropout_prob)
        # self.LayerNorm = nn.LayerNorm(embedding_dim)

    def forward(self, user_emb, topk_emb):
        # Query: 原始用户, Key/Value: Top-K相似用户
        q = self.query(user_emb).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        k = self.key(topk_emb)  # [batch_size, k, embedding_dim]
        v = self.value(topk_emb)  # [batch_size, k, embedding_dim]

        attention_scores = torch.matmul(q, k.transpose(
            -2, -1)) / (self.embedding_dim**0.5)
        p_attn = F.softmax(attention_scores, dim=-1)

        # p_attn = self.attn_dropout(p_attn)
        weighted_sum = torch.matmul(p_attn, v).squeeze(1)
        # weighted_sum = self.LayerNorm(weighted_sum)
        return weighted_sum


class TransMatch_EX(nn.Module):

    def __init__(self,
                 conf,
                 u_topk_IJs,
                 i_topk_UJs,
                 j_topk_UIs,
                 neighbor_params=None,
                 visual_features=None,
                 item_cate=None):
        super().__init__()
        self.visual_features = visual_features
        self.use_context = conf['context']
        self.use_path = conf['path']
        self.hidden_dim = conf['hidden_dim']
        self.user_num = conf['user_num']
        self.item_num = conf['item_num']
        self.batch_size = conf['batch_size']
        self.device = conf['device']
        self.score_type = conf['score_type']
        self.use_Nor = conf['use_Nor']
        self.topk_i = conf['topk_i']
        self.use_hard_neg = conf['use_hard_neg']
        self.context_enhance = conf['context_enhance']
        self.neighbor_params = neighbor_params
        self.entity2edges = torch.LongTensor(self.neighbor_params[0]).to(
            self.device)
        self.edge2entities = torch.LongTensor(self.neighbor_params[1]).to(
            self.device)
        self.edge2relation = torch.LongTensor(self.neighbor_params[2]).to(
            self.device)
        self.relation2entities = torch.LongTensor(self.neighbor_params[3]).to(
            self.device)
        self.entity2edge_set = self.neighbor_params[4]
        self.relation2entity_set = self.neighbor_params[5]
        self.u_topk_IJs = u_topk_IJs.to(self.device)
        self.i_topk_UJs = i_topk_UJs.to(self.device)
        self.j_topk_UIs = j_topk_UIs.to(self.device)

        self.pretrain_mode = conf['pretrain_mode']
        self.pretrain_layer_num = conf['pretrain_layer_num']
        self.use_pretrain = conf['use_pretrain']

        self.use_selfatt = conf['use_selfatt']
        self.topk_u = conf['topk_u']
        self.topk_i = conf['topk_i']
        self.self_attention = Attention(self.hidden_dim)
        self.self_attention_v = Attention(self.hidden_dim)
        self.pretrained_model = conf['pretrained_model']

        self.self_attention_u = Attention(self.hidden_dim)
        self.self_attention_v_u = Attention(self.hidden_dim)

        # self.pretrain_model_file = f"{conf['pretrained_model']}.pth.tar"
        # self.pretrain_model_dir = 'model/iqon_s/pretrained_model/'

        if conf['dataset'] == "iqon_s":
            pretrain_model_file = f"{conf['pretrained_model']}_AUC_0.6809.pth.tar"
        elif conf['dataset'] == "Polyvore_519":
            if conf['mode'] == 'RB':
                pretrain_model_file = f"{conf['pretrained_model']}.pth.tar" #"epoch_118_p0c0_RB_AUC_0.7645.pth"
        elif conf['dataset'] == "IQON3000":
            if conf['mode'] == 'RB':
                pretrain_model_file = f"{conf['pretrained_model']}.pth.tar" #"epoch_77_p0c0_RB_AUC_0.8607.pth"

        pretrain_model_dir = './saved/' + conf['dataset'] + '/pretrained_model/'
        self.pretrain_model_path = os.path.join(pretrain_model_dir, pretrain_model_file)


        # self.pretrain_model_path = os.path.join(self.pretrain_model_dir, self.pretrain_model_file)
        if self.pretrain_mode or not self.use_pretrain: 
            if self.pretrained_model == 'TransR':
                self.transe = TransR(conf,
                                     visual_features=self.visual_features)
            elif self.pretrained_model == 'TransE':
                self.transe = TransE(conf,
                                     visual_features=self.visual_features)
            # elif self.pretrained_model == 'TransMatch_ori':
                # self.transe = TransMatch(conf, neighbor_params = self.neighbor_params,
                #                      visual_features=self.visual_features)

        elif os.path.exists(self.pretrain_model_path) and self.use_pretrain:
            # if self.pretrained_model == 'TransE':
            #     self.transe = TransMatch(conf, neighbor_params = self.neighbor_params,
            #                          visual_features=self.visual_features)
            # checkpoint = torch.load(self.pretrain_model_path)
            # self.transe.load_state_dict(checkpoint['state_dict'], strict=False)

            self.transe = torch.load(self.pretrain_model_path)
            print('Continuing training with existing model...')   
            print(self.transe)     

        # self.all_bottoms_id = self._get_all_bottoms_id()
        self.margin = nn.Parameter(torch.tensor(0.7))
        self.temp = nn.Parameter(torch.tensor(0.07))

        if self.pretrain_mode:
            self.use_context = False
            self.use_path = False
        else:
            self.use_context = True
            self.use_path = False

        if self.use_context:
            self.context_hops = conf['context_hops']
            # define aggregators for each layer
            self.agg_param = conf['agg_param']
            self.neighbor_samples = conf['neighbor_samples']
            if conf['neighbor_agg'] == 'mean':
                self.neighbor_agg = MeanAggregator
            elif conf['neighbor_agg'] == 'concat':
                self.neighbor_agg = ConcatAggregator

            self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())
            if self.score_type == 'mlp':
                self.scorer = nn.Linear(self.hidden_dim, 1)

        if self.use_path:
            self.path_weight = conf['path_weight']
            self.path_agg = conf['path_agg']
            self.path_enhance = conf['path_enhance']

    # def _get_all_bottoms_id(self):
    #     train_df = pd.read_csv("data/iqon_s/train.csv", header=None).astype('int')
    #     train_df.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    #     test_df = pd.read_csv("data/iqon_s/test.csv", header=None).astype('int')
    #     test_df.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    #     valid_df = pd.read_csv("data/iqon_s/val.csv", header=None).astype('int')
    #     valid_df.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    #     all_bottoms_id = pd.concat([train_df["pos_bottom_idx"], test_df["pos_bottom_idx"], valid_df["pos_bottom_idx"],
    #         train_df["neg_bottom_idx"], test_df["neg_bottom_idx"], valid_df["neg_bottom_idx"]], ignore_index=True).unique()
    #     return torch.LongTensor(all_bottoms_id).to(self.device)

    def find_topk_js_for_ui(self, U_latent, I_latent, J_bias, J_latent,
                            all_embs):
        # all_bottoms_embs = self.transe.i_embeddings_i(self.all_bottoms_id)
        # all_bottoms_feas_v = self.transe.visual_features(self.all_bottoms_id)
        # 剔除positive ks -> 'j'的索引
        mask_positive = torch.all(
            all_embs.unsqueeze(1) != J_latent.unsqueeze(0), dim=-1).all(dim=-1)
        mask_input = torch.all(all_embs.unsqueeze(1) != I_latent.unsqueeze(0),
                               dim=-1).all(dim=-1)
        mask = mask_positive & mask_input
        masked_j_embs = all_embs[mask]

        U_latent = U_latent.unsqueeze(1).expand(-1, masked_j_embs.size(0),
                                                -1)  #bs, all_bottoms, hd
        I_latent = I_latent.unsqueeze(1).expand(-1, masked_j_embs.size(0), -1)
        J_bias = J_bias.unsqueeze(1).expand(-1, masked_j_embs.size(0),
                                            -1)  #bs, all_bottoms, 1
        masked_j_embs = masked_j_embs.unsqueeze(0).expand(
            U_latent.size(0), -1, -1)  #bs, all_bottoms, hd
        # distances = J_bias - torch.sum(torch.pow((U_latent + I_latent - masked_j_embss), 2), -1, keepdim=True)
        distances = J_bias.squeeze(-1) - torch.norm(
            U_latent + I_latent - masked_j_embs, dim=-1)  #bs, all_bottoms
        topk_scores, topk_indices = torch.topk(distances, self.topk_i,
                                               dim=-1)  #bs, topk
        return topk_scores  #, topk_indices

    def _get_neighbor_aggregators(self):
        aggregators = []  # store all aggregators
        for i in range(self.context_hops):
            aggregators.append(
                self.neighbor_agg(emb_dim=self.hidden_dim,
                                  self_included=True,
                                  agg_param=self.agg_param))
        return aggregators

    def _get_entity_neighbors_and_masks(self,
                                        relations,
                                        entity_pairs,
                                        train_edges=None):
        bs = relations.size(0)
        edges_list = [relations]
        nodes_list = []
        masks = []
        if train_edges != None:
            train_edges = torch.unsqueeze(train_edges, -1)  # [batch_size, 1]
        if len(entity_pairs.size()) == 2:  #training case
            mode = 'train'
            self_entities = entity_pairs.unsqueeze(-1).expand(
                -1, -1, self.neighbor_samples).reshape(bs, -1)
        elif len(entity_pairs.size()) == 3:  #inference case
            mode = 'test'
            candi_num = entity_pairs.size(1)
            self_entities = entity_pairs.unsqueeze(-1).expand(
                -1, -1, -1, self.neighbor_samples).reshape(bs, candi_num, -1)
        for i in range(self.context_hops):
            if i == 0:
                neighbor_entities = entity_pairs  # [bs, 2] in training, [bs, 100, 2] in inference
                nodes_list.append(neighbor_entities)
                if mode == 'train':
                    neighbor_edges = torch.index_select(
                        self.entity2edges, 0,
                        neighbor_entities.view(-1)).view([bs, -1])
                elif mode == 'test':
                    neighbor_edges = torch.index_select(
                        self.entity2edges, 0,
                        neighbor_entities.view(-1)).view([bs, candi_num, -1])
            else:
                if mode == 'train':
                    neighbor_entities = torch.index_select(
                        self.edge2entities, 0,
                        edges_list[-1].view(-1)).reshape([bs, -1,
                                                          2])  #bs, -1, 2
                    nodes_list.append(neighbor_entities)
                    neighbor_edges = torch.index_select(
                        self.entity2edges, 0,
                        neighbor_entities.view(-1)).view([bs, -1])

                elif mode == 'test':
                    neighbor_entities = torch.index_select(
                        self.edge2entities, 0,
                        edges_list[-1].view(-1)).reshape(
                            [bs, candi_num, -1, 2])
                    nodes_list.append(neighbor_entities)
                    neighbor_edges = torch.index_select(
                        self.entity2edges, 0,
                        neighbor_entities.view(-1)).view([bs, candi_num, -1])

            edges_list.append(neighbor_edges)
            if train_edges != None:
                mask = neighbor_edges - train_edges  # [batch_size, -1]
            else:
                mask = neighbor_edges
            mask = (mask != 0).float()
            masks.append(mask)
        # edge_list: [[bs,], [bs, 10], [bs, 100], ...]

        return edges_list, nodes_list, masks

    def _aggregate_neighbors_train(self,
                                   edge_list,
                                   entity_list,
                                   mask_list,
                                   relation_features,
                                   entity_features,
                                   visual=False):
        bs = edge_list[0].size(0)
        if visual:
            entity_vectors = [
                self.transe.visual_nn_comp(entity_features[entity_list[0]])
            ]
        else:
            entity_vectors = [entity_features(entity_list[0])]
        edge_vectors = [relation_features(edge_list[0])
                        ]  # bs, candi_num, emb_dim
        for edges in edge_list[
                1:]:  # len(edge_list) = self.context_hops+1, len(entity_list) = self.context_hops
            relations = torch.index_select(self.edge2relation, 0,
                                           edges.view(-1)).view(
                                               list(edges.shape))  #
            edge_vectors.append(relation_features(relations))

        for entities in entity_list[1:]:
            if visual:
                entity_vectors.append(
                    self.transe.visual_nn_comp(entity_features[entities]))
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
            neighbor_edge_shape = [
                bs, -1, 2, self.neighbor_samples, aggregator.emb_dim
            ]
            neighbor_entity_shape = [bs, -1, 2, aggregator.emb_dim]
            masks_shape = [bs, -1, 2, self.neighbor_samples, 1]
            for hop in range(self.context_hops -
                             i):  # aggregate in inverse order
                self_edge_vectors, neighbor_entity_vectors = aggregator(
                    self_vectors=edge_vectors[hop],
                    neighbor_entity_vectors=entity_vectors[hop].view(
                        neighbor_entity_shape),
                    neighbor_edge_vectors=edge_vectors[hop + 1].view(
                        neighbor_edge_shape),
                    masks=mask_list[hop].view(masks_shape))

                edge_vectors_next_iter.append(self_edge_vectors)
                node_vectors_next_iter.append(neighbor_entity_vectors)

            edge_vectors = edge_vectors_next_iter
            entity_vectors = node_vectors_next_iter
        # edge_vectos[0]: [self.batch_size, 1, self.n_relations]


#         res = edge_vectors[0].view([bs, self.n_relations])
        return edge_vectors[0], entity_vectors[0].squeeze(1)

    def _aggregate_neighbors_test(self,
                                  edge_list,
                                  entity_list,
                                  mask_list,
                                  relation_features,
                                  entity_features,
                                  visual=False):
        """A list of masks, indicating the presence or absence of edges in the
        context."""
        bs, candi_num = edge_list[0].size()
        if visual:
            entity_vectors = [
                self.transe.visual_nn_comp(entity_features[entity_list[0]])
            ]
        else:
            entity_vectors = [entity_features(entity_list[0])]
        edge_vectors = [relation_features(edge_list[0])
                        ]  # bs, candi_num, emb_dim
        for edges in edge_list[
                1:]:  # len(edge_list) = self.context_hops+1, len(entity_list) = self.context_hops
            relations = torch.index_select(self.edge2relation, 0,
                                           edges.view(-1)).view(
                                               list(edges.shape))  #
            edge_vectors.append(
                relation_features(relations)
            )  # initial state: edge vectors are pure relation features
        for entities in entity_list[1:]:
            if visual:
                entity_vectors.append(
                    self.transe.visual_nn_comp(entity_features[entities]))
            else:
                entity_vectors.append(entity_features(entities))

        for i in range(self.context_hops):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter, node_vectors_next_iter = [], []
            neighbor_edge_shape = [
                bs, candi_num, -1, 2, self.neighbor_samples, aggregator.emb_dim
            ]
            neighbor_entity_shape = [bs, candi_num, -1, 2, aggregator.emb_dim]
            masks_shape = [bs, candi_num, -1, 2, self.neighbor_samples, 1]
            for hop in range(self.context_hops -
                             i):  # aggregate in inverse order
                self_edge_vectors, neighbor_entity_vectors = aggregator(
                    self_vectors=edge_vectors[hop],
                    neighbor_entity_vectors=entity_vectors[hop].view(
                        neighbor_entity_shape),
                    neighbor_edge_vectors=edge_vectors[hop + 1].view(
                        neighbor_edge_shape),
                    masks=mask_list[hop].view(masks_shape))
                edge_vectors_next_iter.append(self_edge_vectors)
                node_vectors_next_iter.append(neighbor_entity_vectors)

            edge_vectors = edge_vectors_next_iter
            entity_vectors = node_vectors_next_iter
        return edge_vectors[0], entity_vectors[0].squeeze(-3)

    def transE_predict(self, u_rep, i_rep, j_rep, j_bias):
        pred = j_bias - torch.sum(
            torch.pow((u_rep + i_rep - j_rep), 2), -1, keepdim=True)
        return pred.squeeze(-1)

    def get_path_rep(self, paths, path_mask, rel_rep):
        # paths: bs, path_num, path_len
        if self.path_agg == 'mean':
            path_rep = torch.mean(
                self.transe.u_embeddings_l(paths) * path_mask.unsqueeze(-1),
                (-2, -3))
        elif self.path_agg == 'sum':
            path_rep = torch.sum(
                self.transe.u_embeddings_l(paths) * path_mask.unsqueeze(-1),
                (-2, -3)) / torch.clamp(torch.sum(path_mask, (-1, -2)),
                                        min=1).unsqueeze(-1)
        elif self.path_agg == 'att':
            # rel_rep: bs, 1， emb_size (train); bs, 2, emb_size (inference)
            path_rep = torch.sum(
                self.transe.u_embeddings_l(paths) * path_mask.unsqueeze(-1),
                dim=-2) / torch.clamp(torch.sum(path_mask, dim=-1),
                                      min=1).unsqueeze(
                                          -1)  # bs, path_num, em_dim
            path_num = paths.size(-2)
            if len(path_rep.size()) == 3:
                rel_path = torch.exp(
                    torch.matmul(rel_rep, path_rep.permute(0, 2, 1))).squeeze(
                        -2)  #bs, path_num
                coef = rel_path / torch.sum(rel_path,
                                            dim=-1).unsqueeze(-1).expand(
                                                -1, path_num)  #bs, path_num
            elif len(path_rep.size()) == 4:
                rel_path = torch.exp(
                    torch.matmul(rel_rep.unsqueeze(-2),
                                 path_rep.permute(0, 1, 3, 2))).squeeze(
                                     -2)  #bs, path_num
                coef = rel_path / torch.sum(
                    rel_path, dim=-1).unsqueeze(-1).expand(-1, -1, path_num)
            path_rep = torch.sum(coef.unsqueeze(-1) * path_rep, dim=-2)
        return path_rep

    def forward(self, batch):
        Us = batch[0]
        Is = batch[1]
        Js = batch[2]
        Ks = batch[3]
        J_bias_v = self.transe.i_bias_v(Js)
        K_bias_v = self.transe.i_bias_v(Ks)

        if self.pretrain_mode:
            R_j, R_k = self.transe.forward(batch)

        else:
            U_latent_ori = self.transe.u_embeddings_l(Us)
            I_latent_ori = self.transe.i_embeddings_i(Is)
            J_latent_ori = self.transe.i_embeddings_i(Js)
            K_latent_ori = self.transe.i_embeddings_i(Ks)

            U_visual_ori = self.transe.u_embeddings_v(Us)
            vis_I_ori = self.visual_features[Is]
            vis_J_ori = self.visual_features[Js]
            vis_K_ori = self.visual_features[Ks]
            I_visual_ori = self.transe.visual_nn_comp(
                vis_I_ori)  #bs, hidden_dim
            J_visual_ori = self.transe.visual_nn_comp(vis_J_ori)
            K_visual_ori = self.transe.visual_nn_comp(vis_K_ori)

            if self.use_context:
                self.entity_pairs_pos = torch.cat(
                    [Is.unsqueeze(1), Js.unsqueeze(1)], dim=-1)
                self.entity_pairs_neg = torch.cat(
                    [Is.unsqueeze(1), Ks.unsqueeze(1)], dim=-1)  # bs, 2
                self.train_edges = batch[4]
                edge_list_pos, entity_list_pos, mask_list_pos = self._get_entity_neighbors_and_masks(
                    Us, self.entity_pairs_pos, self.train_edges)
                edge_pos_rep, entity_pos_rep = self._aggregate_neighbors_train(
                    edge_list_pos, entity_list_pos, mask_list_pos,
                    self.transe.u_embeddings_l, self.transe.i_embeddings_i)
                # edge_rep: bs, emb_dim; entity_rep: bs, 2, emb_dim
                # edge_rep can be directly used to predict the score
                edge_list_neg, entity_list_neg, mask_list_neg = self._get_entity_neighbors_and_masks(
                    Us, self.entity_pairs_neg)
                edge_neg_rep, entity_neg_rep = self._aggregate_neighbors_train(
                    edge_list_neg, entity_list_neg, mask_list_neg,
                    self.transe.u_embeddings_l, self.transe.i_embeddings_i)
                U_latent_pos = edge_pos_rep
                U_latent_neg = edge_neg_rep
                I_latent_pos = entity_pos_rep[:, 0, :]
                J_latent = entity_pos_rep[:, 1, :]
                I_latent_neg = entity_neg_rep[:, 0, :]
                K_latent = entity_neg_rep[:, 1, :]

                if self.score_type == 'mlp':
                    R_j = self.scorer(U_latent_pos).squeeze(-1)
                    R_k = self.scorer(U_latent_neg).squeeze(
                        -1)  #or apply classification loss
                elif self.score_type == 'transE':
                    J_bias_l = self.transe.i_bias_l(Js)
                    K_bias_l = self.transe.i_bias_l(Ks)
                    R_j = self.transE_predict(U_latent_pos.squeeze(-2),
                                              I_latent_pos, J_latent, J_bias_l)
                    R_k = self.transE_predict(U_latent_neg.squeeze(-2),
                                              I_latent_neg, K_latent, K_bias_l)

                edge_pos_rep_v, entity_pos_rep_v = self._aggregate_neighbors_train(
                    edge_list_pos, entity_list_pos, mask_list_pos,
                    self.transe.u_embeddings_v, self.visual_features, True)
                edge_neg_rep_v, entity_neg_rep_v = self._aggregate_neighbors_train(
                    edge_list_neg, entity_list_neg, mask_list_neg,
                    self.transe.u_embeddings_v, self.visual_features, True)

                if self.score_type == 'mlp':
                    R_j += self.scorer(edge_pos_rep_v).squeeze(-1)
                    R_k += self.scorer(edge_neg_rep_v).squeeze(
                        -1)  #or apply ification loss
                elif self.score_type == 'transE':
                    R_j_v = self.transE_predict(edge_pos_rep_v.squeeze(-2),
                                                entity_pos_rep_v[:, 0, :],
                                                entity_pos_rep_v[:, 1, :],
                                                J_bias_v)
                    R_k_v = self.transE_predict(edge_neg_rep_v.squeeze(-2),
                                                entity_neg_rep_v[:, 0, :],
                                                entity_neg_rep_v[:, 1, :],
                                                K_bias_v)
                    R_j += R_j_v
                    R_k += R_k_v

            if self.context_enhance:
                U_latent = self._group_topkIJs_for_U_(
                    self.transe.i_embeddings_i, Us, U_latent_ori,
                    self.self_attention_u)
                I_latent = self._group_topkUJs_for_I_(
                    self.transe.i_embeddings_i, Is, I_latent_ori,
                    self.self_attention_u)
                J_latent = self._group_topkUJs_for_I_(
                    self.transe.i_embeddings_i, Js, J_latent_ori,
                    self.self_attention_u)
                K_latent = self._group_topkUJs_for_I_(
                    self.transe.i_embeddings_i, Ks, K_latent_ori,
                    self.self_attention_u)
                if self.use_Nor:
                    U_latent = F.normalize(U_latent, dim=0)
                    I_latent = F.normalize(I_latent, dim=0)
                    J_latent = F.normalize(J_latent, dim=0)
                    K_latent = F.normalize(K_latent, dim=0)
                if self.score_type == 'mlp':
                    edge_pos_rep = torch.cat([U_latent, I_latent, J_latent],
                                             dim=-1)
                    edge_neg_rep = torch.cat([U_latent, I_latent, K_latent],
                                             dim=-1)
                    edge_pos_rep = self.layer(
                        edge_pos_rep)  # [bs, -1, emb_dim]
                    edge_pos_rep = F.relu(edge_pos_rep)
                    edge_neg_rep = self.layer(
                        edge_neg_rep)  # [bs, -1, emb_dim]
                    edge_neg_rep = F.relu(edge_neg_rep)
                    R_j_c = self.scorer(edge_pos_rep).squeeze(-1)
                    R_k_c = self.scorer(edge_neg_rep).squeeze(-1)
                elif self.score_type == 'transE':
                    J_bias_l = self.transe.i_bias_l(Js)
                    K_bias_l = self.transe.i_bias_l(Ks)
                    if self.pretrained_model == 'transR':
                        projection_matrix = self.transe.projection_matrix(
                            Us).view(Us.size(0), self.hidden_dim,
                                     self.hidden_dim).transpose(1, 2)
                        I_latent = torch.matmul(I_latent.unsqueeze(1),
                                                projection_matrix).squeeze(1)
                        J_latent = torch.matmul(J_latent.unsqueeze(1),
                                                projection_matrix).squeeze(1)
                        K_latent = torch.matmul(K_latent.unsqueeze(1),
                                                projection_matrix).squeeze(1)
                    R_j_c = self.transE_predict(U_latent, I_latent, J_latent,
                                                J_bias_l)
                    R_k_c = self.transE_predict(U_latent, I_latent, K_latent,
                                                K_bias_l)
                if self.use_context:
                    R_j += 0.8* R_j_c
                    R_k += 0.8* R_k_c
                else:
                    R_j = R_j_c
                    R_k = R_k_c

            if not self.use_context and not self.context_enhance:  # original TransE
                U_latent, I_latent, J_latent, K_latent = U_latent_ori, I_latent_ori, J_latent_ori, K_latent_ori
                if self.score_type == 'mlp':
                    edge_pos_rep = torch.cat([U_latent, I_latent, J_latent],
                                             dim=-1)
                    edge_neg_rep = torch.cat([U_latent, I_latent, K_latent],
                                             dim=-1)
                    edge_pos_rep = self.layer(
                        edge_pos_rep)  # [bs, -1, emb_dim]
                    edge_pos_rep = F.relu(edge_pos_rep)
                    edge_neg_rep = self.layer(
                        edge_neg_rep)  # [bs, -1, emb_dim]
                    edge_neg_rep = F.relu(edge_neg_rep)
                    R_j = self.scorer(edge_pos_rep).squeeze(-1)
                    R_k = self.scorer(edge_neg_rep).squeeze(-1)

                elif self.score_type == 'transE':
                    J_bias_l = self.transe.i_bias_l(Js)
                    K_bias_l = self.transe.i_bias_l(Ks)

                    if self.pretrained_model == 'transR':
                        projection_matrix = self.transe.projection_matrix(
                            Us).view(Us.size(0), self.hidden_dim,
                                     self.hidden_dim).transpose(1, 2)
                        I_latent = torch.matmul(I_latent.unsqueeze(1),
                                                projection_matrix).squeeze(1)
                        J_latent = torch.matmul(J_latent.unsqueeze(1),
                                                projection_matrix).squeeze(1)
                        K_latent = torch.matmul(K_latent.unsqueeze(1),
                                                projection_matrix).squeeze(1)
                    R_j = self.transE_predict(U_latent, I_latent, J_latent,
                                              J_bias_l)
                    R_k = self.transE_predict(U_latent, I_latent, K_latent,
                                              K_bias_l)

            if self.context_enhance:
                all_item_latent_v = self.transe.visual_nn_comp(
                    self.visual_features)
                U_visual = self._group_topkIJs_for_U_v_(
                    all_item_latent_v, Us, U_visual_ori,
                    self.self_attention_v_u)
                I_visual = self._group_topkUJs_for_I_v_(
                    all_item_latent_v, Is, I_visual_ori,
                    self.self_attention_v_u)
                J_visual = self._group_topkUJs_for_I_v_(
                    all_item_latent_v, Js, J_visual_ori,
                    self.self_attention_v_u)
                K_visual = self._group_topkUJs_for_I_v_(
                    all_item_latent_v, Ks, K_visual_ori,
                    self.self_attention_v_u)
                # K_visual = K_visual_ori
                if self.use_Nor:
                    U_visual = F.normalize(U_visual, dim=0)
                    I_visual = F.normalize(I_visual, dim=0)
                    J_visual = F.normalize(J_visual, dim=0)
                    K_visual = F.normalize(K_visual, dim=0)
                # if self.use_selfatt:
                #     all_item_latent_v = self.transe.visual_nn_comp(self.visual_features)
                #     topk_users_idxs = self.find_topk_similar_users(U_visual_ori, self.transe.u_embeddings_v.weight.clone().detach(), self.topk_u)
                #     topk_Is_idxs = self.find_topk_similar_users(I_visual_ori, all_item_latent_v, self.topk_i)
                #     topk_Js_idxs = self.find_topk_similar_users(J_visual_ori, all_item_latent_v, self.topk_i)
                #     topk_Ks_idxs = self.find_topk_similar_users(K_visual_ori, all_item_latent_v, self.topk_i)
                #     U_visual = self.aggregate_item_embeddings(self.transe.u_embeddings_v, topk_users_idxs, U_visual_ori, self.self_attention_v_u)
                #     I_visual = self.aggregate_embeddings(all_item_latent_v, topk_Is_idxs, I_visual_ori, self.self_attention_v)
                #     J_visual = self.aggregate_embeddings(all_item_latent_v, topk_Js_idxs, J_visual_ori, self.self_attention_v)
                #     K_visual = self.aggregate_embeddings(all_item_latent_v, topk_Ks_idxs, K_visual_ori, self.self_attention_v)
                #     if self.use_Nor:
                #         U_visual = F.normalize(U_visual,dim=0)
                #         I_visual = F.normalize(I_visual,dim=0)
                #         J_visual = F.normalize(J_visual,dim=0)
                #         K_visual = F.normalize(K_visual,dim=0)
                #     else:
                #         U_visual, I_visual, J_visual, K_visual = U_visual, I_visual, J_visual, K_visual
                # elif not self.context_enhance and not self.use_selfatt:
                if self.score_type == 'mlp':
                    pass
                elif self.score_type == 'transE':
                    if self.pretrained_model == 'transR':
                        projection_matrix_v = self.transe.projection_matrix_v(
                            Us).view(Us.size(0), self.hidden_dim,
                                     self.hidden_dim).transpose(1, 2)
                        I_visual = torch.matmul(I_visual.unsqueeze(1),
                                                projection_matrix_v).squeeze(1)
                        J_visual = torch.matmul(J_visual.unsqueeze(1),
                                                projection_matrix_v).squeeze(1)
                        K_visual = torch.matmul(K_visual.unsqueeze(1),
                                                projection_matrix_v).squeeze(1)
                    R_j_v = self.transE_predict(U_visual, I_visual, J_visual,
                                                J_bias_v)
                    R_k_v = self.transE_predict(U_visual, I_visual, K_visual,
                                                K_bias_v)
                    R_j += 0.8*R_j_v
                    R_k += 0.8*R_k_v

            if not self.use_context and not self.context_enhance:  # original TransE
                U_visual, I_visual, J_visual, K_visual = U_visual_ori, I_visual_ori, J_visual_ori, K_visual_ori
                if self.score_type == 'mlp':
                    pass
                elif self.score_type == 'transE':
                    if self.pretrained_model == 'transR':
                        projection_matrix_v = self.transe.projection_matrix_v(
                            Us).view(Us.size(0), self.hidden_dim,
                                     self.hidden_dim).transpose(1, 2)
                        I_visual = torch.matmul(I_visual.unsqueeze(1),
                                                projection_matrix_v).squeeze(1)
                        J_visual = torch.matmul(J_visual.unsqueeze(1),
                                                projection_matrix_v).squeeze(1)
                        K_visual = torch.matmul(K_visual.unsqueeze(1),
                                                projection_matrix_v).squeeze(1)
                    R_j_v = self.transE_predict(U_visual, I_visual, J_visual,
                                                J_bias_v)
                    R_k_v = self.transE_predict(U_visual, I_visual, K_visual,
                                                K_bias_v)
                    R_j += R_j_v
                    R_k += R_k_v

            if self.use_hard_neg:
                all_embs = self.transe.i_embeddings_i.weight.clone().detach()
                all_feas_v = self.transe.visual_nn_comp(
                    self.transe.visual_features)
                # all_bottoms_embs = self.transe.i_embeddings_i(self.all_bottoms_id)
                # all_bottoms_feas_v = self.transe.visual_features(self.all_bottoms_id)
                topk_js_scores = self.find_topk_js_for_ui(
                    U_latent_ori, I_latent_ori, J_bias_l, J_latent_ori,
                    all_embs)
                pos_score = self.transE_predict(U_latent_ori, I_latent_ori,
                                                J_latent_ori, J_bias_l)
                true_neg_score = self.transE_predict(U_latent_ori,
                                                     I_latent_ori,
                                                     K_latent_ori, K_bias_l)
                topk_js_scores += self.find_topk_js_for_ui(
                    U_visual_ori, I_visual_ori, J_bias_v, J_visual_ori,
                    all_feas_v)
                pos_score += self.transE_predict(U_visual_ori, I_visual_ori,
                                                 J_visual_ori, J_bias_v)
                true_neg_score += self.transE_predict(U_visual_ori,
                                                      I_visual_ori,
                                                      K_visual_ori, K_bias_v)
                # Compute CL loss with hard negatives
                cl_loss = self.contrastive_loss_1(pos_score, topk_js_scores,
                                                  self.margin)
                # cl_loss = self.hard_contrastive_loss(pos_score, topk_js_scores, true_neg_score, self.temp)
                # cl_loss = self.contrastive_loss(pos_score, topk_js_scores, self.temp)

            if self.use_path:
                pos_paths = batch[5]
                neg_paths = batch[6]
                pos_path_mask = batch[7]
                neg_path_mask = batch[8]

                pos_path_rep = self.get_path_rep(
                    pos_paths, pos_path_mask,
                    U_latent_pos)  # bs, path_num, path_len
                neg_path_rep = self.get_path_rep(
                    neg_paths, neg_path_mask,
                    U_latent_neg)  # bs, path_num, path_len

                R_j_p = self.transE_predict(pos_path_rep, I_latent_pos,
                                            J_latent, J_bias_l)
                R_k_p = self.transE_predict(neg_path_rep, I_latent_neg,
                                            K_latent, K_bias_l)

                R_j += R_j_p * self.path_weight
                R_k += R_k_p * self.path_weight

        loss = bpr_loss(R_j, R_k)  #original bpr loss
        if self.use_hard_neg:
            return 0.8 * loss + 0.2 * cl_loss
        else:
            return loss

    def contrastive_loss_1(self, pos_score, neg_scores, margin):
        positive_loss = torch.mean((pos_score - margin)**2)
        negative_loss = torch.mean(F.relu(margin - neg_scores)**2, dim=1)
        # negative_loss = torch.sum(F.relu(margin - neg_scores) ** 2) #not work
        negative_loss = torch.mean(negative_loss)
        total_loss = positive_loss + negative_loss
        return total_loss

    def contrastive_loss(self, pos_score, neg_scores, temp):
        exp_pos_score = torch.exp(pos_score / temp)
        positive_loss = -torch.log(exp_pos_score / torch.sum(exp_pos_score))
        exp_neg_scores = torch.exp(neg_scores / temp)
        negative_loss = -torch.log(
            torch.sum(exp_neg_scores, dim=-1) / torch.sum(exp_neg_scores))
        total_loss = torch.mean(positive_loss + negative_loss)
        return total_loss

    def hard_contrastive_loss(self, pos_score, hard_neg_scores, true_neg_score,
                              temp):
        exp_pos_score = torch.exp(pos_score / temp)
        exp_hard_neg_scores = torch.sum(torch.exp(true_neg_score / temp),
                                        dim=-1)
        exp_true_neg_score = torch.exp(true_neg_score / temp)
        ttl_score = exp_pos_score + exp_hard_neg_scores + exp_true_neg_score
        total_loss = -torch.mean(torch.log(exp_pos_score / ttl_score))
        return total_loss

    def cal_c_loss(self, pos, neg, anchor=None):
        # pos: [batch_size, pos_path_num, emb_size]
        # aug: [batch_size, neg_path_num, emb_size]
        if anchor is not None:
            pos_score = torch.matmul(anchor, pos.permute(0, 2, 1))
            neg_score = torch.matmul(anchor, neg.permute(0, 2, 1))
        else:
            pos_score = torch.matmul(pos, pos.permute(
                0, 2, 1))  # bs, pos_path_num, pos_path_num
            neg_score = torch.matmul(pos, neg.permute(
                0, 2, 1))  # bs, pos_path_num, neg_path_num
        pos_score = torch.sum(torch.exp(pos_score / self.c_temp),
                              dim=-1)  # bs, pos_num
        neg_score = torch.sum(torch.exp(neg_score / self.c_temp),
                              dim=-1)  # bs, pos_num
        ttl_score = pos_score + neg_score
        c_loss = -torch.mean(torch.log(pos_score / ttl_score))
        return c_loss

    def find_topk_similar_users(self, batch_user_embeddings,
                                all_user_embeddings, k):
        similarity = torch.matmul(batch_user_embeddings,
                                  all_user_embeddings.t())
        # 取Top-K，不包括自己
        topk_values, topk_indices = torch.topk(similarity,
                                               k=k + 1,
                                               largest=True,
                                               sorted=True)
        self_indices = torch.arange(batch_user_embeddings.size(0)).unsqueeze(1)
        self_indices = self_indices.to(self.device)
        topk_indices = topk_indices.to(self.device)
        topk_indices = topk_indices.where(topk_indices != self_indices,
                                          topk_indices[:, -1].unsqueeze(1))
        return topk_indices[:, 1:].to(self.device)

    def aggregate_item_embeddings(self, item_embeddings, topk_indices,
                                  batch_item_embeddings, att_model):
        topk_item_embeddings = item_embeddings(topk_indices)
        aggregated_embeddings = att_model(batch_item_embeddings,
                                          topk_item_embeddings)
        return aggregated_embeddings

    def aggregate_embeddings(self, user_embeddings, topk_indices,
                             batch_user_embeddings, att_model_u):
        topk_user_embeddings = user_embeddings[topk_indices]
        aggregated_embeddings = att_model_u(batch_user_embeddings,
                                            topk_user_embeddings)
        return aggregated_embeddings

    def _mean_emb_(self, embs, topk_embs):
        embs_expanded = embs.unsqueeze(1).expand(-1, topk_embs.size(1), -1)
        concatenated_tensor = torch.cat((topk_embs, embs_expanded), dim=2)
        average_tensor = torch.mean(concatenated_tensor, dim=1)
        return average_tensor

    def _group_topkIJs_for_U_(self, item_embeddings, Us, batch_user_embeddings,
                              att_model_u):
        topk_Is = self.u_topk_IJs[Us]  # bs, 2, topk_u
        topk_Is = topk_Is[:, 0, :self.topk_u]  # bs, topk_u
        topk_Is_embeddings = item_embeddings(topk_Is)  #bs, topk_u, hd
        topk_Js = self.u_topk_IJs[Us]
        topk_Js = topk_Js[:, 1, :self.topk_u]
        topk_Js_embeddings = item_embeddings(topk_Js)  #bs, topk_u, hd
        combined_tensor = torch.cat((topk_Is_embeddings, topk_Js_embeddings),
                                    dim=1)  #bs, topk_u*2, hd
        # aggregated_embeddings = att_model_u(batch_user_embeddings, combined_tensor)
        # aggregated_embeddings = self._mean_emb_(batch_user_embeddings, combined_tensor)
        aggregated_embeddings = batch_user_embeddings + torch.mean(
            combined_tensor, dim=1
        ) * self.agg_param  #.view([batch_user_embeddings.size(1), self.emb_dim])
        return aggregated_embeddings

    def _group_topkIJs_for_U_v_(self, item_embeddings, Us,
                                batch_user_embeddings, att_model_u):
        topk_Is = self.u_topk_IJs[Us]  # bs, 2, topk_u
        topk_Is = topk_Is[:, 0, :self.topk_u]  # bs, topk_u
        topk_Is_embeddings = item_embeddings[topk_Is.long()]  #bs, topk_u, hd # torch.Size([1024, 3, 32])
        topk_Js = self.u_topk_IJs[Us]
        topk_Js = topk_Js[:, 1, :self.topk_u]
        topk_Js_embeddings = item_embeddings[topk_Js.long()]  #bs, topk_u, hd
        combined_tensor = torch.cat((topk_Is_embeddings, topk_Js_embeddings),
                                    dim=1)  #bs, topk_u*2, hd
        # aggregated_embeddings = att_model_u(batch_user_embeddings, combined_tensor)
        # aggregated_embeddings = self._mean_emb_(batch_user_embeddings, combined_tensor)
        aggregated_embeddings = batch_user_embeddings + torch.mean(
            combined_tensor, dim=1) * self.agg_param
        return aggregated_embeddings

    def _group_topkUJs_for_I_(self, item_embeddings, Is, batch_Is_embeddings,
                              att_model_u):
        topk_UJs = self.i_topk_UJs[Is]
        topk_Us = topk_UJs[:, 0, :self.topk_i]  # bs, 2, topk_u -> bs, topk_u
        topk_Us_embeddings = self.transe.u_embeddings_l(
            topk_Us)  #bs, topk_u, hd
        topk_Js = topk_UJs[:, 1, :self.topk_i]
        topk_Js_embeddings = item_embeddings(topk_Js.long())  #bs, topk_u, hd
        combined_tensor = torch.cat((topk_Us_embeddings, topk_Js_embeddings),
                                    dim=1)  #bs, topk_u*2, hd
        # aggregated_embeddings = att_model_u(batch_Is_embeddings, combined_tensor)
        # aggregated_embeddings = self._mean_emb_(batch_Is_embeddings, combined_tensor)
        aggregated_embeddings = batch_Is_embeddings + torch.mean(
            combined_tensor, dim=1) * self.agg_param
        return aggregated_embeddings

    def _group_topkUJs_for_I_v_(self, item_embeddings, Is, batch_Is_embeddings,
                                att_model_u):
        topk_UJs = self.i_topk_UJs[Is]
        topk_Us = topk_UJs[:, 0, :self.topk_i]  # bs, 2, topk_u -> bs, topk_u
        topk_Us_embeddings = self.transe.u_embeddings_v(
            topk_Us)  #bs, topk_u, hd
        topk_Js = topk_UJs[:, 1, :self.topk_i]
        topk_Js_embeddings = item_embeddings[topk_Js.long()]  #bs, topk_u, hd
        combined_tensor = torch.cat((topk_Us_embeddings, topk_Js_embeddings),
                                    dim=1)  #bs, topk_u*2, hd
        # aggregated_embeddings = att_model_u(batch_Is_embeddings, combined_tensor)
        # aggregated_embeddings = self._mean_emb_(batch_Is_embeddings, combined_tensor)
        aggregated_embeddings = batch_Is_embeddings + torch.mean(
            combined_tensor, dim=1) * self.agg_param
        return aggregated_embeddings

    def inference(self, batch):
        Us = batch[0]
        Is = batch[1]
        Js = batch[2]
        Ks = batch[3]
        J_list = torch.cat([Js.unsqueeze(1), Ks], dim=-1)
        j_num = J_list.size(1)
        Us_exp = Us.unsqueeze(1).expand(-1, j_num)  #bs, j_num
        Is_exp = Is.unsqueeze(1).expand(-1, j_num)
        J_bias_l = self.transe.i_bias_l(J_list)

        if self.pretrain_mode:
            scores = self.transe.inference(batch)
            # print("pretaining!!!")

        else:
            U_latent_ori = self.transe.u_embeddings_l(Us)
            I_latent_ori = self.transe.i_embeddings_i(Is)
            J_latent_ori = self.transe.i_embeddings_i(Js)
            K_latent_ori = self.transe.i_embeddings_i(Ks.squeeze(1))
            J_bias_l = self.transe.i_bias_l(J_list)
            U_visual_ori = self.transe.u_embeddings_v(Us)
            vis_I = self.visual_features[Is]
            vis_J = self.visual_features[Js]
            vis_K = self.visual_features[Ks.squeeze(1)]
            I_visual_ori = self.transe.visual_nn_comp(vis_I)  #bs, hidden_dim
            J_visual_ori = self.transe.visual_nn_comp(vis_J)
            K_visual_ori = self.transe.visual_nn_comp(vis_K)
            J_bias_v = self.transe.i_bias_v(J_list)

            if self.use_context:
                self.entity_pairs = torch.cat(
                    [Is_exp.unsqueeze(-1),
                     J_list.unsqueeze(-1)], dim=-1)  # bs, j_num, 2
                edge_list, entity_list, mask_list = self._get_entity_neighbors_and_masks(
                    Us_exp, self.entity_pairs)
                edge_rep, entity_rep = self._aggregate_neighbors_test(
                    edge_list, entity_list, mask_list,
                    self.transe.u_embeddings_l, self.transe.i_embeddings_i)
                U_latent = edge_rep.squeeze(-2)
                I_latent_ii = entity_rep[:, :, 0, :]
                Js_latent_ii = entity_rep[:, :, 1, :]
                if self.score_type == 'mlp':
                    scores = self.scorer(edge_rep).squeeze(-1)
                elif self.score_type == 'transE':
                    scores = self.transE_predict(U_latent, I_latent_ii,
                                                 Js_latent_ii, J_bias_l)

                edge_rep_v, entity_rep_v = self._aggregate_neighbors_test(
                    edge_list, entity_list, mask_list,
                    self.transe.u_embeddings_v, self.visual_features, True)
                U_visual = edge_rep_v.squeeze(-2)
                I_visual_ii = entity_rep_v[:, :, 0, :]
                Js_visual_ii = entity_rep_v[:, :, 1, :]
                if self.score_type == 'mlp':
                    scores += self.scorer(edge_rep_v).squeeze(-1)
                elif self.score_type == 'transE':
                    scores += self.transE_predict(U_visual, I_visual_ii,
                                                  Js_visual_ii, J_bias_v)

            if self.context_enhance:
                U_latent = self._group_topkIJs_for_U_(
                    self.transe.i_embeddings_i, Us, U_latent_ori,
                    self.self_attention_u)
                I_latent = self._group_topkUJs_for_I_(
                    self.transe.i_embeddings_i, Is, I_latent_ori,
                    self.self_attention_u)
                J_latent = self._group_topkUJs_for_I_(
                    self.transe.i_embeddings_i, Js, J_latent_ori,
                    self.self_attention_u)
                K_latent = self._group_topkUJs_for_I_(
                    self.transe.i_embeddings_i, Ks.squeeze(1), K_latent_ori,
                    self.self_attention_u)
                # K_latent = K_latent_ori
                if self.use_Nor:
                    U_latent = F.normalize(U_latent, dim=0)
                    I_latent = F.normalize(I_latent, dim=0)
                    J_latent = F.normalize(J_latent, dim=0)
                    K_latent = F.normalize(K_latent, dim=0)

                U_latent = U_latent.unsqueeze(1).expand(-1, j_num, -1)
                I_latent = I_latent.unsqueeze(1).expand(-1, j_num, -1)
                Js_latent_ii = torch.stack((J_latent, K_latent), dim=1)
                if self.score_type == 'mlp':
                    edge_rep = torch.cat([U_latent, I_latent, Js_latent_ii],
                                         dim=-1)
                    edge_rep = self.layer(edge_rep)  # [bs, -1, emb_dim]
                    edge_rep = F.relu(edge_rep)
                    score_c += self.scorer(edge_rep).squeeze(-1)
                    if self.use_context:
                        scores += 0.8*scores_c
                    else:
                        scores = scores_c

                elif self.score_type == 'transE':
                    if self.pretrained_model == 'transR':
                        projection_matrix = self.transe.projection_matrix(
                            Us_exp).view(Us_exp.size(0), Us_exp.size(1),
                                         self.hidden_dim,
                                         self.hidden_dim).transpose(2, 3)
                        I_latent = torch.matmul(I_latent.unsqueeze(2),
                                                projection_matrix).squeeze(2)
                        Js_latent_ii = torch.matmul(
                            Js_latent_ii.unsqueeze(2),
                            projection_matrix).squeeze(2)
                    scores_c = self.transE_predict(U_latent, I_latent,
                                                   Js_latent_ii, J_bias_l)
                    if self.use_context:
                        scores += scores_c
                    else:
                        scores = scores_c
            if not self.use_context and not self.context_enhance:  # original TransE
                U_latent, I_latent, J_latent, K_latent = U_latent_ori, I_latent_ori, J_latent_ori, K_latent_ori
                U_latent = U_latent.unsqueeze(1).expand(-1, j_num, -1)
                I_latent = I_latent.unsqueeze(1).expand(-1, j_num, -1)
                Js_latent_ii = torch.stack((J_latent, K_latent), dim=1)
                if self.score_type == 'mlp':
                    edge_rep = torch.cat([U_latent, I_latent, Js_latent_ii],
                                         dim=-1)
                    edge_rep = self.layer(edge_rep)  # [bs, -1, emb_dim]
                    edge_rep = F.relu(edge_rep)
                    scores = self.scorer(edge_rep).squeeze(-1)
                elif self.score_type == 'transE':
                    if self.pretrained_model == 'transR':
                        projection_matrix = self.transe.projection_matrix(
                            Us_exp).view(Us_exp.size(0), Us_exp.size(1),
                                         self.hidden_dim,
                                         self.hidden_dim).transpose(2, 3)
                        I_latent = torch.matmul(I_latent.unsqueeze(2),
                                                projection_matrix).squeeze(2)
                        Js_latent_ii = torch.matmul(
                            Js_latent_ii.unsqueeze(2),
                            projection_matrix).squeeze(2)
                    scores = self.transE_predict(U_latent, I_latent,
                                                 Js_latent_ii, J_bias_l)

            if self.context_enhance:
                all_item_latent_v = self.transe.visual_nn_comp(
                    self.visual_features)
                U_visual = self._group_topkIJs_for_U_v_(
                    all_item_latent_v, Us, U_visual_ori,
                    self.self_attention_v_u)
                I_visual = self._group_topkUJs_for_I_v_(
                    all_item_latent_v, Is, I_visual_ori,
                    self.self_attention_v_u)
                J_visual = self._group_topkUJs_for_I_v_(
                    all_item_latent_v, Js, J_visual_ori,
                    self.self_attention_v_u)
                K_visual = self._group_topkUJs_for_I_v_(
                    all_item_latent_v, Ks.squeeze(1), K_visual_ori,
                    self.self_attention_v_u)
                # K_visual = K_visual_ori
                if self.use_Nor:
                    U_visual = F.normalize(U_visual, dim=0)
                    I_visual = F.normalize(I_visual, dim=0)
                    J_visual = F.normalize(J_visual, dim=0)
                    K_visual = F.normalize(K_visual, dim=0)
                U_visual = U_visual.unsqueeze(1).expand(-1, j_num, -1)
                I_visual = I_visual.unsqueeze(1).expand(-1, j_num, -1)
                Js_visual_ii = torch.stack((J_visual, K_visual), dim=1)
                if self.pretrained_model == 'transR':
                    projection_matrix_v = self.transe.projection_matrix_v(
                        Us_exp).view(Us_exp.size(0), Us_exp.size(1),
                                     self.hidden_dim,
                                     self.hidden_dim).transpose(2, 3)
                    I_visual = torch.matmul(I_visual.unsqueeze(2),
                                            projection_matrix_v).squeeze(2)
                    Js_visual_ii = torch.matmul(Js_visual_ii.unsqueeze(2),
                                                projection_matrix_v).squeeze(2)
                scores += 0.8*self.transE_predict(U_visual, I_visual, Js_visual_ii,
                                              J_bias_v)

            if not self.use_context and not self.context_enhance:  # original TransE
                U_visual, I_visual, J_visual, K_visual = U_visual_ori, I_visual_ori, J_visual_ori, K_visual_ori
                U_visual = U_visual.unsqueeze(1).expand(-1, j_num, -1)
                I_visual = I_visual.unsqueeze(1).expand(-1, j_num, -1)
                Js_visual_ii = torch.stack((J_visual, K_visual), dim=1)
                if self.pretrained_model == 'transR':
                    projection_matrix_v = self.transe.projection_matrix_v(
                        Us_exp).view(Us_exp.size(0), Us_exp.size(1),
                                     self.hidden_dim,
                                     self.hidden_dim).transpose(2, 3)
                    I_visual = torch.matmul(I_visual.unsqueeze(2),
                                            projection_matrix_v).squeeze(2)
                    Js_visual_ii = torch.matmul(Js_visual_ii.unsqueeze(2),
                                                projection_matrix_v).squeeze(2)
                scores += self.transE_predict(U_visual, I_visual, Js_visual_ii,
                                              J_bias_v)

            if self.use_path:
                j_paths = batch[4]
                k_paths = batch[5]
                j_path_mask = batch[6]
                k_path_mask = batch[7]

                if len(k_paths.size()) == 3:
                    jk_paths = torch.cat(
                        [j_paths.unsqueeze(1),
                         k_paths.unsqueeze(1)],
                        dim=1)  # [bs, 2, path_num, path_len]
                    jk_path_mask = torch.cat(
                        [j_path_mask.unsqueeze(1),
                         k_path_mask.unsqueeze(1)],
                        dim=1)  #

                elif len(k_paths.size()) == 4:
                    jk_paths = torch.cat([j_paths.unsqueeze(1), k_paths],
                                         dim=1)
                    jk_path_mask = torch.cat(
                        [j_path_mask.unsqueeze(1), k_path_mask], dim=1)  #

                path_rep = self.get_path_rep(jk_paths, jk_path_mask, U_latent)
                scores += self.path_weight * self.transE_predict(path_rep, I_latent_ii,
                                              Js_latent_ii, J_bias_l)

        return scores
