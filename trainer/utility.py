import csv
import json
import os
import pdb
from datetime import datetime
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from PIL import Image
from torchvision import transforms
from trainer.loader_CP import *
from tool.kg_utils import *


def load_csv_data(train_data_path):
    result = []
    with open(train_data_path) as fp:
        for line in fp:
            t = line.strip().split(',')
            t = [int(i) for i in t]
            result.append(t)
    return result


def reindex_polyvore_features(visual_features_ori, item_map, conf):
    name_item_map = json.load(
        open(conf['new_datapath'] + '/item_name_map.json'))
    item_name_map = {}
    for i_name, item in name_item_map.items():
        item_name_map[item] = i_name.split('.')[0]
    visual_features = []

    iid_item_map = {}
    for item, iid in item_map.items():
        iid_item_map[iid] = item

    for iid in range(len(item_map)):
        item = iid_item_map[iid]
        i_name = item_name_map[item]
        visual_fea = visual_features_ori[i_name]
        visual_features.append(torch.Tensor(visual_fea))
    torch.save(torch.stack(visual_features, dim=0),
               conf['new_datapath'] + conf['visual_features_tensor'])
    return visual_features


def reindex_iqon_features(visual_features_ori, item_map, device):
    visual_features = []
    text_features = []
    id_item_map = {}
    no_fea_item = 0

    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    image_path = '/mnt/juanjuan/cafidata/outfit_datasets/IQON3000_imgs/'
    img_size = 224
    image_transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for item in item_map:
        id_item_map[item_map[item]] = item

    fea_size = 2048
    for iid in range(len(id_item_map)):
        item = str(id_item_map[iid])
        if str(item) in visual_features_ori:
            visual_fea = visual_features_ori[str(item)]
        else:
            try:
                img = Image.open(image_path + '%s_m.jpg' %
                                 (item)).convert('RGB')
                img = image_transform(img).unsqueeze(0)
                visual_fea = model(img.to(device)).squeeze(-1).squeeze(
                    -1).squeeze(0).detach().cpu().tolist()
            except Exception:
                no_fea_item += 1
                visual_fea = [0] * fea_size
        visual_features.append(torch.Tensor(visual_fea))

    print('no feature item: %d' % no_fea_item)
    return visual_features


def map_item_cates(conf, item_cates_ori, cate_items_ori, item_map):
    item_cates = {}
    cate_items = {}
    missing_items = set()
    cate_map = {}
    cate_id = 0
    for item, iid in item_map.items():
        if str(item) not in item_cates_ori:
            missing_items.add(item)
            cate = -1
        else:
            cate = item_cates_ori[str(item)]

        if cate not in cate_map:
            cate_map[cate] = cate_id
            cate_id += 1

        cid = cate_map[cate]
        item_cates[iid] = cid
        if cid not in cate_items:
            cate_items[cid] = []
        cate_items[cid].append(iid)
    json.dump(cate_items, open(conf['new_datapath'] + conf['cate_items'], 'w'))
    json.dump(item_cates, open(conf['new_datapath'] + conf['item_cates'], 'w'))
    json.dump(cate_map, open(conf['new_datapath'] + 'cate_map.json', 'w'))
    return item_cates, cate_items, cate_map


def index_interaction_data(train_data, test_data, val_data, conf):
    item_cates_ori = json.load(
        open(conf['new_datapath'] + conf['item_cates_ori']))
    cate_items_ori = json.load(
        open(conf['new_datapath'] + conf['cate_items_ori']))
    user_set = set()
    item_set = set()
    for data in [train_data, test_data, val_data]:
        for one in data:
            u, i, j, k = one
            user_set.add(u)
            item_set.add(i)
            item_set.add(j)
            item_set.add(k)

    user_map = {}
    for cnt, u in enumerate(user_set):
        user_map[u] = cnt
    item_map = {}
    for cnt, i in enumerate(item_set):
        item_map[i] = cnt

    print('Indexing training ...')
    new_train, new_test, new_val = [], [], []
    for one in train_data:
        u, i, j, k = one
        new_one = [user_map[u], item_map[i], item_map[j], item_map[k]]
        new_train.append(new_one)
    print('Indexing testing ...')
    for one in test_data:
        u, i, j, k = one
        new_one = [user_map[u], item_map[i], item_map[j], item_map[k]]
        new_test.append(new_one)
    print('Indexing validation ...')
    print('All data indexed!')

    for one in val_data:
        u, i, j, k = one
        new_one = [user_map[u], item_map[i], item_map[j], item_map[k]]
        new_val.append(new_one)

    if conf['save_new_data']:
        train_f = open(conf['new_datapath'] + 'train.csv', 'w')
        train_writer = csv.writer(train_f)
        test_f = open(conf['new_datapath'] + 'test.csv', 'w')
        test_writer = csv.writer(test_f)
        val_f = open(conf['new_datapath'] + 'val.csv', 'w')
        val_writer = csv.writer(val_f)
        for new_one in new_train:
            train_writer.writerow(new_one)
        for new_one in new_test:
            test_writer.writerow(new_one)
        for new_one in new_val:
            val_writer.writerow(new_one)
        json.dump(user_map, open(conf['new_datapath'] + conf['user_map'], 'w'))
        json.dump(item_map, open(conf['new_datapath'] + conf['item_map'], 'w'))
        print('New data saved!')
    return user_map, item_map, new_train, new_test, new_val, item_cates_ori, cate_items_ori


def get_U_topk_IJs_tensor(u_topk_IJs):
    tensor_list = []
    for key, value in u_topk_IJs.items():
        tensor = torch.tensor(value, dtype=torch.int32)
        tensor_list.append(tensor)
    stacked_tensor = torch.stack(tensor_list)
    return stacked_tensor


def get_fake_triplets(conf, path, topk_u, topk_i, u_ijs, i_ujs, j_uis):
    fake_triplets = []
    for key, value in u_ijs.items():
        for i in range(topk_u):
            triplet = tuple([int(key), value[0][i], value[1][i]])  # u, i, j
            fake_triplets.append(triplet)
    train_df = pd.read_csv(conf['train_data'], header=None).astype('int')
    train_df.columns = ['user_idx', 'top_idx', 'pos_bottom_idx', 'neg_bottom_idx']
    unique_fake_triplets = train_df[['user_idx', 'top_idx', 'pos_bottom_idx']].drop_duplicates().values.tolist()
    unique_neg_triplets = train_df[['user_idx', 'top_idx', 'neg_bottom_idx']].drop_duplicates().values.tolist()   
    # incase we add a nagetive sample to the triplets                              
    for triplet in fake_triplets:
        if triplet not in unique_fake_triplets:
            if triplet not in unique_neg_triplets:
                unique_fake_triplets.append(triplet)
    with open(path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for triplet in unique_fake_triplets:
            csv_writer.writerow(triplet)

    return unique_fake_triplets


def prepare_data(conf):
    print('preparing data ...')
    if conf['dataset'] == 'IQON3000':
        if conf['mode'] == 'RB':
            conf['train_data'] = conf['root_datapath'] + conf['dataset'] + '/data/train_indexed.csv'
            conf['valid_data'] = conf['root_datapath'] + conf['dataset'] + '/data/valid_indexed.csv'
            conf['test_data'] = conf['root_datapath'] + conf['dataset'] + '/data/test_indexed.csv'
        elif conf['mode'] == 'RT':
            conf['train_data'] = conf['root_datapath'] + conf['dataset'] + '/data/train_indexed_top.csv'
            conf['valid_data'] = conf['root_datapath'] + conf['dataset'] + '/data/valid_indexed_top.csv'
            conf['test_data'] = conf['root_datapath'] + conf['dataset'] + '/data/test_indexed_top.csv'
        print(conf['train_data'])
        train_data = load_csv_data(conf['train_data'] )
        test_data = load_csv_data(conf['valid_data'])
        val_data = load_csv_data(conf['test_data'])
        visual_path = conf['root_datapath'] + conf['dataset'] + '/feat/' + conf['visual_features_tensor']
        visual_features = torch.load(visual_path,map_location=lambda a, b: a.cpu())
        print('visual features loaded !')
        user_map = json.load(open('./dataset/IQON3000/data/user_map.json'))
        item_map = json.load(open('./dataset/IQON3000/data/item_map.json'))
        # visual_features = reindex_iqon_features(visual_features_ori, item_map, conf['device'])

    elif conf['dataset'] == 'Polyvore_519': #original ,not subsampled 
        if conf['mode'] == 'RB':
            conf['train_data'] = conf['root_datapath'] + conf['dataset'] + '/polyvore_U_519_data/train_data.csv' #'/polyvore_U_519_subset_data/train_sub_data.csv'
            conf['valid_data'] = conf['root_datapath'] + conf['dataset'] + '/polyvore_U_519_data/valid_data.csv' #'/polyvore_U_519_subset_data/valid_sub_data.csv'
            conf['test_data'] = conf['root_datapath'] + conf['dataset'] + '/polyvore_U_519_data/test_data.csv' #'/polyvore_U_519_subset_data/test_sub_data.csv'
        elif conf['mode'] == 'RT':
            conf['train_data'] = conf['root_datapath'] + conf['dataset'] + '/polyvore_U_519_data/train_data_RT.csv' 
            conf['valid_data'] = conf['root_datapath'] + conf['dataset'] + '/polyvore_U_519_data/valid_data_RT.csv' 
            conf['test_data'] = conf['root_datapath'] + conf['dataset'] + '/polyvore_U_519_data/test_data_RT.csv' 
            print(os.path.exists(conf['train_data']))
            if not os.path.exists(conf['train_data']):
                train_path_RB = conf['root_datapath'] + conf['dataset'] + '/polyvore_U_519_data/train_data.csv' 
                val_path_RB = conf['root_datapath'] + conf['dataset'] + '/polyvore_U_519_data/valid_data.csv'
                test_path_RB = conf['root_datapath'] + conf['dataset'] + '/polyvore_U_519_data/test_data.csv'
                all_bottoms, all_tops = all_candidates(train_path_RB, val_path_RB, test_path_RB)
                new_train_RT= create_RT_data(train_path_RB, all_tops)
                new_valid_RT = create_RT_data(val_path_RB, all_tops)
                new_test_RT = create_RT_data(test_path_RB, all_tops)
                with open(conf['train_data'], mode='w', newline='') as file:
                    writer = csv.writer(file)
                    for row in new_train_RT:
                        writer.writerow(row)
                with open(conf['valid_data'], mode='w', newline='') as file:
                    writer = csv.writer(file)
                    for row in new_valid_RT:
                        writer.writerow(row)
                with open(conf['test_data'], mode='w', newline='') as file:
                    writer = csv.writer(file)
                    for row in new_test_RT:
                        writer.writerow(row)

        print(conf['train_data'])
        train_data = load_csv_data(conf['train_data'])
        test_data = load_csv_data(conf['valid_data'])
        val_data = load_csv_data(conf['test_data'])
        visual_path = conf['root_datapath'] + conf['dataset'] + '/polyvore_U_519_data/tb_visual_fea_tensor_new'
        visual_features = torch.load(visual_path,map_location=lambda a, b: a.cpu())
        print('visual features loaded !')
        user_map = json.load(open('./dataset/Polyvore_519/polyvore_U_519_data/user_map'))
        item_map = json.load(open('./dataset/Polyvore_519/polyvore_U_519_data/item_map'))

    elif conf['dataset'] == 'iqon_s':
        conf['train_data'] = conf['new_datapath'] + '/UII_train_quadruple.json'
        conf['valid_data'] = conf['new_datapath'] + '/UII_test_quadruple.json'
        conf['test_data'] = conf['new_datapath'] + '/UII_valid_quadruple.json'
        train_data = json.load(open(conf['train_data']))
        test_data = json.load(open(conf['valid_data']))
        val_data = json.load(open(conf['test_data']))
        user_map, item_map, new_train, new_test, new_val, item_cates_ori, cate_items_ori = index_interaction_data(
        train_data, test_data, val_data, conf)
        print('training, testing, validation data loaded !')
        # visual_features = torch.stack(visual_features, dim=0)
        item_cates, cate_items, cate_map = map_item_cates(conf, item_cates_ori,
                                                        cate_items_ori, item_map)
        print('category information done !')
        return new_train, new_test, new_val, visual_features, user_map, item_map , item_cates, cate_items
    all_bottoms, all_tops = all_candidates(conf['train_data'], conf['test_data'], conf['valid_data'])
    item_cates = {"0": all_bottoms}
    return train_data, test_data, val_data, visual_features, user_map, item_map, all_bottoms, all_tops, item_cates


def load_cache(conf):
    new_train = load_csv_data(conf['new_datapath'] + 'train.csv')
    new_test = load_csv_data(conf['new_datapath'] + 'test.csv')
    new_val = load_csv_data(conf['new_datapath'] + 'val.csv')
    visual_features = torch.load(conf['new_datapath'] + conf['visual_features_tensor'], map_location=lambda a, b: a.cpu())
    item_cates = json.load(open(conf['new_datapath'] + conf['item_cates']))
    cate_items = json.load(open(conf['new_datapath'] + conf['cate_items']))
    user_map = json.load(open(conf['new_datapath'] + conf['user_map']))
    item_map = json.load(open(conf['new_datapath'] + conf['item_map']))
    return new_train, new_test, new_val, visual_features, user_map, item_map, item_cates, cate_items

# def prepare_wide_evaluate(test_data, neg_num, item_cates, cate_items):
#     wide_test_list = []
#     for data in test_data:
#         u, i, j, neg_j = data
#         neg_list = [neg_j]
#         j_cate = item_cates[str(j)]
#         candi_js = cate_items[str(j_cate)]
#         if len(candi_js) < neg_num:
#             continue
#         while len(neg_list) < neg_num:
#             neg_j = random.choice(candi_js)
#             while neg_j == j or neg_j in neg_list:
#                 neg_j = random.choice(candi_js)
#             neg_list.append(neg_j)
#         test_list = [u, i, j] + neg_list
#         wide_test_list.append(test_list)
#     return np.array(wide_test_list)  # neg_num + 1

def all_candidates(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path, header=None).astype('int')
    test_df = pd.read_csv(val_path, header=None).astype('int')
    valid_df = pd.read_csv(test_path, header=None).astype('int')
    all_df = pd.concat([train_df, test_df, valid_df], axis=0)
    all_df.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    all_bottoms_dict = all_df["pos_bottom_idx"].value_counts().to_dict()
    all_bottoms = list(all_bottoms_dict.keys())
    all_tops_dict = all_df["top_idx"].value_counts().to_dict()
    all_tops = list(all_tops_dict.keys())
    return all_bottoms, all_tops # rank based on popularity

def create_RT_data(train_data_path, all_tops):
    result = []
    with open(train_data_path) as fp:
        for line in fp:
            t = line.strip().split(',')
            t = [int(i) for i in t]
            u,i,j,k = t         
            candi_is = [item for item in all_tops if item != i]           
            neg_i = random.choice(candi_is)
            new_t = [u,j,i,int(neg_i)]         
            result.append(new_t)        
    return result


def prepare_wide_evaluate(all_bottoms, test_data, neg_num):
    wide_test_list = []
    for data in test_data:
        u, i, j, neg_j = data
        neg_list = [neg_j]
        available_bottoms = [item for item in all_bottoms if item != j]
        if len(available_bottoms) < neg_num -1:
            raise ValueError("Not enough items to sample from after excluding pos_bottom_idx")
        neg_js = random.sample(available_bottoms, neg_num -1)
        neg_list = neg_list + neg_js
        test_list = [u, i, j] + neg_list
        wide_test_list.append(test_list)
        # print(test_list) #[327, 16888, 39973, 43714, 58795, 80205, 66063, 102180, ...]
    return np.array(wide_test_list)


class Dataset():
    def __init__(self, conf):
        self.conf = conf
        dataconf = yaml.safe_load(open('./config/CP_config.yaml'))
        dataconf['dataset'] = self.conf['dataset']
        if self.conf['dataset'] == 'IQON3000':
            dataconf['new_datapath'] = dataconf['root_datapath'] + 'IQON3000/'
        elif self.conf['dataset'] == 'Polyvore_519':
            dataconf['new_datapath'] = dataconf['root_datapath'] + 'Polyvore_519/'
        elif self.conf['dataset'] == 'iqon_s':
            dataconf['new_datapath'] = dataconf['root_datapath'] + 'iqon_s/'
        elif self.conf['dataset'] == 'ifashion':
            dataconf['new_datapath'] = dataconf['root_datapath'] + 'iFashion/'
        dataconf['device'] = self.conf['device']
        if not os.path.exists(dataconf['new_datapath']):
            os.makedirs(dataconf['new_datapath'])

        # if conf['data_status'] == 'prepare_save':
        #     dataconf['save_new_data'] = 1
        #     train_data, test_data, val_data, visual_features, self.user_map, self.item_map, self.item_cates, self.cate_items = prepare_data(dataconf)
        # elif conf['data_status'] == 'prepare_no_save':
        #     dataconf['save_new_data'] = 0
        #     train_data, test_data, val_data, visual_features, self.user_map, self.item_map, self.item_cates, self.cate_items = prepare_data(dataconf)
        # elif conf['data_status'] == 'use_old':
        #     train_data, test_data, val_data, visual_features, self.user_map, self.item_map, self.item_cates, self.cate_items = load_cache(dataconf)

        train_data, test_data, val_data, visual_features, self.user_map, self.item_map, self.all_bottoms, self.all_tops, self.item_cates = prepare_data(dataconf)
        

        if conf['wide_evaluate']:
            try:
                test_data_L = np.load(dataconf['new_datapath'] +'/test_data_%d.npy' % (self.conf['neg_num']))
                print(test_data_L.shpae())
                val_data_L = np.load(dataconf['new_datapath'] +'/val_data_%d.npy' % (self.conf['neg_num']))
                print('test and validation data for top %d evaluation loaded' %self.conf['topk'][0])
            except Exception:
                print('preparing test and validation data for top %d evaluation'% conf['topk'][0])
                test_data_L = prepare_wide_evaluate(self.all_bottoms, test_data, self.conf['neg_num'])
                val_data_L = prepare_wide_evaluate(self.all_bottoms, val_data, self.conf['neg_num'])
                np.save(dataconf['new_datapath'] + '/test_data_%d.npy' %(self.conf['neg_num']), test_data_L)
                np.save(dataconf['new_datapath'] + '/val_data_%d.npy' %(self.conf['neg_num']), val_data_L)

        self.visual_features = torch.cat((visual_features, torch.zeros(1, 2048)), 0)
        self.train_items, self.train_users = self.get_train_user_items(train_data)
        self.new_items = set(set(self.item_map.values()) - self.train_items)
        self.new_users = set(set(self.user_map.values()) - self.train_users)

        if conf['model'] == 'TransMatch':
            if conf['path_enhance']:
                u_IJS_path = dataconf['new_datapath'] + dataconf['u_topk_IJs']
                i_UJS_path = dataconf['new_datapath'] + dataconf['i_topk_UJs']
                j_UIS_path = dataconf['new_datapath'] + dataconf['j_topk_UIs']
                u_topk_IJs = json.load(open(u_IJS_path))
                i_topk_UJs = json.load(open(i_UJS_path))
                j_topk_UIs = json.load(open(j_UIS_path))
                self.u_topk_IJs = get_U_topk_IJs_tensor(u_topk_IJs)
                self.i_topk_UJs = get_U_topk_IJs_tensor(i_topk_UJs)
                self.j_topk_UIs = get_U_topk_IJs_tensor(j_topk_UIs)

                fake_triplets_path = dataconf[
                    'new_datapath'] + 'fake_triplets_U%d_I%d.csv' % (self.conf['topk_u'], self.conf['topk_i'])
                if os.path.exists(fake_triplets_path):
                    self.unique_fake_triplets = load_csv_data(fake_triplets_path)
                else:
                    unique_fake_triplets = get_fake_triplets(fake_triplets_path, self.conf['topk_u'], self.conf['topk_i'], u_topk_IJs, i_topk_UJs, j_topk_UIs)
                    self.unique_fake_triplets = load_csv_data(fake_triplets_path)
                entity2edge_set, edge2entities, edge2relation, e2re, relation2entity_set = build_kg_topk(self.unique_fake_triplets)
            else:
                entity2edge_set, edge2entities, edge2relation, e2re, relation2entity_set = build_kg(train_data)
            self.null_entity = len(self.item_map)
            self.null_relation = len(self.user_map)
            self.null_edge = len(edge2entities)
            edge2entities.append([self.null_entity, self.null_entity])
            edge2relation.append(self.null_relation)
            entity2edges = []
            relation2entities = []
            for i in range(len(self.item_map) + 1):
                if i not in entity2edge_set:
                    entity2edge_set[i] = {self.null_edge}
                sampled_neighbors = np.random.choice(list(entity2edge_set[i]),
                    size=self.conf['neighbor_samples'],
                    replace=len(entity2edge_set[i]) < self.conf['neighbor_samples'])
                entity2edges.append(sampled_neighbors)

            for u in range(len(self.user_map) + 1):
                if u not in relation2entity_set:
                    relation2entities.append([self.null_relation] * self.conf['neighbor_samples'])
                    continue

                entities = relation2entity_set[u]
                sampled_neighbors = np.random.choice(entities, size=self.conf['neighbor_samples'], replace=len(entities) < self.conf['neighbor_samples'])
                relation2entities.append(sampled_neighbors)
            self.neighbor_params = [
                np.array(entity2edges),
                np.array(edge2entities),
                np.array(edge2relation),
                np.array(relation2entities), entity2edge_set,
                relation2entity_set
            ]

            if self.conf['path']:
                # if applying pathcon_path or path_contrastive, all conditional paths in the kg should be obtained first.
                if self.conf['path_enhance']:
                    ht_dict_path = dataconf[
                        'new_datapath'] + '/ht_dict_topk_U%d_I%d_%d_%d.json' % (
                            self.conf['topk_u'], self.conf['topk_i'], self.conf['path_num'],
                            self.conf['max_path_len'])
                    path_tensor_path = dataconf[
                        'new_datapath'] + '/path_topk_U%d_I%d_%d_%d_all' % (
                            self.conf['topk_u'], self.conf['topk_i'], self.conf['path_num'],
                            self.conf['max_path_len'])
                else:
                    ht_dict_path = dataconf[
                        'new_datapath'] + '/ht_dict_%d_%d.json' % (
                            self.conf['path_num'], self.conf['max_path_len'])
                    path_tensor_path = dataconf[
                        'new_datapath'] + '/path_%d_%d_all' % (
                            self.conf['path_num'], self.conf['max_path_len'])
                if not os.path.exists(ht_dict_path) or not os.path.exists(
                        path_tensor_path):
                    print('generating shorter than %d paths ...' %
                          conf['max_path_len'])
                    if self.conf['path_enhance']:
                        head2tails = get_h2t_topk(self.unique_fake_triplets,
                                                  test_data, val_data)
                    else:
                        head2tails = get_h2t(train_data, test_data, val_data)
                    head2tails_list = [(k, v) for k, v in head2tails.items()]
                    n_cores, pool, range_list = get_params_for_mp(
                        len(head2tails_list))
                    results = pool.map(
                        count_all_paths,
                        zip([e2re] * n_cores, [self.conf['max_path_len']] * n_cores,
                            [head2tails_list[i[0]:i[1]] for i in range_list],
                            # [self.item_cates] * n_cores, range(n_cores)))
                            range(n_cores)))

                    res = defaultdict(set)
                    for ht2paths in results:
                        res.update(ht2paths)
                    new_res = defaultdict(set)
                    ht_num = 0
                    path_tensor = []
                    ht_dict = {}
                    for ht, paths in res.items():
                        if len(paths) > self.conf['path_num']:
                            paths = list(paths)
                            if len(np.shape(paths)) > 1:
                                p_len = len(paths[0])
                                if p_len == 1:
                                    paths.append(
                                        [
                                            self.null_relation,
                                            self.null_relation
                                        ]
                                    )  # when all paths have same length, the array turns into two-dimentional, this operation is to fix this bug
                                else:
                                    paths.append([self.null_relation])
                            try:
                                pick_paths = np.random.choice(
                                    paths, self.conf['path_num'])
                            except Exception:
                                continue
                                # pdb.set_trace()
                            new_paths = []
                            for path in pick_paths:
                                path = list(path)
                                while len(path) < self.conf['max_path_len']:
                                    path.append(self.null_relation)
                                new_paths.append(path)
                        else:
                            pick_paths = list(paths)
                            new_paths = []
                            for path in pick_paths:
                                path = list(path)
                                while len(path) < self.conf['max_path_len']:
                                    path.append(self.null_relation)
                                new_paths.append(path)
                            while len(new_paths) < self.conf['path_num']:
                                new_paths.append([self.null_relation] *
                                                 self.conf['max_path_len'])

                        ht_str = str(ht[0]) + '_' + str(ht[1])
                        path_tensor.append(new_paths)
                        ht_dict[ht_str] = ht_num
                        ht_num += 1

                    self.ht2paths = path_tensor
                    json.dump(ht_dict, open(ht_dict_path, 'w'))
                    torch.save(self.ht2paths, path_tensor_path)

                else:
                    ht_dict = json.load(open(ht_dict_path))
                    self.ht2paths = torch.load(path_tensor_path)

        if self.conf['path']:
            self.traindata = Train_Data_Path(train_data, self.all_bottoms,
                                             ht_dict,
                                             self.ht2paths, e2re,
                                             self.conf['max_path_len'],
                                             self.conf['path_num'],
                                             self.null_relation)

            self.testdata = Test_Data_Path(test_data, ht_dict, self.ht2paths,
                                           e2re, self.conf['max_path_len'],
                                           self.conf['path_num'],
                                           self.null_relation)
            self.valdata = Test_Data_Path(val_data, ht_dict, self.ht2paths,
                                          e2re, self.conf['max_path_len'],
                                          self.conf['path_num'], self.null_relation)
        else:
            self.traindata = Train_Data(train_data, self.all_bottoms)
            self.testdata = Test_Data(test_data)
            self.valdata = Test_Data(val_data)

        self.train_loader = DataLoader(self.traindata,
                                       batch_size=self.conf['batch_size'],
                                       shuffle=True)
        self.test_loader = DataLoader(self.testdata,
                                      batch_size=self.conf['batch_size'],
                                      shuffle=False)
        self.val_loader = DataLoader(self.valdata,
                                     batch_size=self.conf['batch_size'],
                                     shuffle=False)
        self.test_loader_list = [self.test_loader, self.val_loader]
        self.test_setting_list = ['test_auc', 'val_auc']

        if self.conf['wide_evaluate']:
            if self.conf['path']:
                self.wide_testdata = Wide_Test_Data_Path(
                    test_data_L, ht_dict, self.ht2paths, e2re,
                    self.conf['max_path_len'], self.conf['path_num'], self.null_relation)
                self.wide_valdata = Wide_Test_Data_Path(
                    val_data_L, ht_dict, self.ht2paths, e2re,
                    self.conf['max_path_len'], self.conf['path_num'], self.null_relation)
            else:
                self.wide_testdata = Wide_Test_Data(test_data_L)
                self.wide_valdata = Wide_Test_Data(val_data_L)
            self.wide_test_loader = DataLoader(
                self.wide_testdata,
                batch_size=self.conf['test_batch_size'],
                shuffle=False)
            self.wide_val_loader = DataLoader(
                self.wide_valdata,
                batch_size=self.conf['test_batch_size'],
                shuffle=False)
            self.test_loader_list += [
                self.wide_test_loader, self.wide_val_loader
            ]
            self.test_setting_list += ['test_topk', 'val_topk']

    def get_train_user_items(self, train_data):
        train_items = set()
        train_users = set()
        for data in train_data:
            u, i, j, _ = data
            train_items.add(j)
            train_items.add(i)
            train_users.add(u)
        return train_items, train_users
