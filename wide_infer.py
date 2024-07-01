import os
import time
import numpy as np
import json
import logging
import argparse
import torch
from torch.nn.functional import logsigmoid
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from util import config
from tool.util import * #AverageMeter, poly_learning_rate, find_free_port, EarlyStopping
from trainer.loader_APCL import Load_Data, Test_Data
import csv
from torch.optim import Adam
from sys import argv
import json
import pdb
from torch.nn import *
import random
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tool.metrics import *
from config.configurator import parse_configure
from Models.BPRs.GPBPR import GPBPR

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def load_embedding_weight(textural_embedding_matrix, device):
    jap2vec = torch.load(textural_embedding_matrix)
    embeding_weight = []
    for jap, vec in jap2vec.items():
        embeding_weight.append(vec.tolist())
    embeding_weight.append(torch.zeros(300))
    embedding_weight = torch.tensor(embeding_weight, device=device)
    return embedding_weight

def interaction_weight(train_data):
    interactions = pd.read_csv(train_data,header=None).astype('int')
    interactions.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    ub_counts = interactions.groupby(["user_idx", "pos_bottom_idx"]).size().reset_index(name='counts')
    ub_counts['inter_weights'] = 1 / np.sqrt(ub_counts['counts'])
    tb_counts = interactions.groupby(["top_idx", "pos_bottom_idx"]).size().reset_index(name='counts')
    tb_counts['inter_weights']  = 1 / np.sqrt(tb_counts['counts'])

    ub_inter_weights_dict = {(int(row['user_idx']), int(row["pos_bottom_idx"])): np.array(row['inter_weights']) for _, row in ub_counts.iterrows()}
    tb_inter_weights_dict = {(int(row["top_idx"]), int(row["pos_bottom_idx"])): np.array(row['inter_weights']) for _, row in tb_counts.iterrows()}
    # for cold-start problems, unseen data in test data, assign defualt median weight
    ub_default_weight = np.median(ub_counts['inter_weights'])
    tb_default_weight = np.median(tb_counts['inter_weights'])
    return ub_inter_weights_dict, tb_inter_weights_dict, ub_default_weight, tb_default_weight 

def Get_Data(train_data_file):
    user_history = pd.read_csv(train_data_file, header=None).astype('int')
    user_history.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    user_bottoms_dict = user_history.groupby("user_idx")["pos_bottom_idx"].agg(list).to_dict()
    user_tops_dict = user_history.groupby("user_idx")["top_idx"].agg(list).to_dict()
    top_bottoms_dict = user_history.groupby("top_idx")["pos_bottom_idx"].agg(list).to_dict()
    popular_bottoms = user_history["pos_bottom_idx"].value_counts().to_dict()
    popular_bottoms = list(popular_bottoms.keys())
    popular_tops = user_history["top_idx"].value_counts().to_dict()
    popular_tops = list(popular_tops.keys())
    popular_users = user_history["user_idx"].value_counts().to_dict()
    popular_users = list(popular_users.keys())
    bottom_user_dict = user_history.groupby("pos_bottom_idx")["user_idx"].agg(list).to_dict()
    return user_bottoms_dict, user_tops_dict, top_bottoms_dict, popular_bottoms, popular_tops, bottom_user_dict, popular_users

def WIDE_INFER(device, model, val_loader, t_len): #for wide infer
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()
    end = time.time()
    pos = 0
    preds = []
    top_scores = []
    ordered_item_ids = []
    with torch.no_grad(): 
        for iteration, aBatch in enumerate(val_loader):
            
            aBatch = [x.to(device) for x in aBatch]
            # itemid= torch.cat([aBatch[2].unsqueeze(1), aBatch[3].expand(-1, args.test_batch_size)], dim=-1)
            itemid= torch.cat([aBatch[2].unsqueeze(1), aBatch[3]], dim=-1)
            scores = model.inference(aBatch, train=False)  #[256, 257]     
            # pos += float(torch.sum(output.ge(0)))
            top_score, tops = torch.topk(scores, k=scores.size(1), dim=-1) #[ 1.3721,  1.2723,  1.2610,  ..., -0.6292, -0.6326, -0.7522], [ 40, 175, 237,  ..., 203,  45, 236],
            preds.append(tops)
            top_scores.append(top_score)
            batch_ordered_item_ids = torch.gather(itemid, 1, tops)
            ordered_item_ids.append(batch_ordered_item_ids)

    preds = torch.cat(preds, dim=0)
    ordered_item_ids = torch.cat(ordered_item_ids, dim=0)
    bs = preds.size(0)
    grd = [0] * bs
    grd_cnt = [1] * bs
    metrics = {}
    for topk in args.k: #args.k:
        metrics[topk] = {}
        REC, MRR, NDCG = get_metrics(grd, grd_cnt, preds.cpu().numpy(), topk)
        metrics[topk]['recall'] = REC
        metrics[topk]['mrr'] = MRR
        metrics[topk]['ndcg'] = NDCG
    # AUC = pos/t_len
    batch_time.update(time.time() - end)
    end = time.time()
    metric_strings = []
    for m in args.metrics:
        for k in args.k:
            metric_strings.append('{}@{}: {:.4f}'.format(m, k, metrics[k][m]))
    logger.info(', '.join(metric_strings))
    return metrics, preds, ordered_item_ids[:, :100]  #, top_scores

def main():
    global logger, writer, args
    # args = get_parser()
    args = parse_configure()
    logger = get_logger()
    
    # args.device = torch.device("cuda:%s"%args.cuda if torch.cuda.is_available() else "cpu")
    # logger.info(args)
    logger.info("=> creating model ...")
    if args.with_visual:
        visual_features_tensor = torch.load(args.visual_features_tensor, map_location= lambda a,b:a.cpu())#torch.Size([142737, 2048])
        v_zeros = torch.zeros(visual_features_tensor.size(-1)).unsqueeze(0)
        visual_features_tensor = torch.cat((visual_features_tensor,v_zeros),0)
        # visual_features_tensor.to(args.device)
    else:
        visual_features_tensor = None 
    if args.with_text:
        text_features_tensor = torch.load(args.textural_features_tensor, map_location= lambda a,b:a.cpu())#torch.Size([142737, 83])       
        t_zeros = torch.zeros(text_features_tensor.size(-1)).unsqueeze(0)
        text_features_tensor = torch.cat((text_features_tensor,t_zeros),0)
        # text_features_tensor.to(args.device)
        if args.dataset == 'IQON3000':
            embedding_weight = load_embedding_weight(args.textural_embedding_matrix, args.device)#torch.Size([54276, 300])
        else:
            embedding_weight = None
    else:
        text_features_tensor = None
        embedding_weight = None

    user_map = json.load(open(args.user_map))
    item_map = json.load(open(args.item_map)) 
    args.user_num = len(user_map)
    args.item_num = len(item_map)

    ub_inter_weights_dict, tb_inter_weights_dict, ub_default_weight, tb_default_weight  = interaction_weight(args.train_data)

    user_bottoms_dict, user_tops_dict, top_bottoms_dict, popular_bottoms, popular_tops, bottom_user_dict, popular_users = Get_Data(args.train_data) 
    test_data_ori = load_csv_data(args.test_data)
    test_data_ori  = torch.LongTensor(test_data_ori)
    test_data = Test_Data(args, test_data_ori, user_bottoms_dict, user_tops_dict, top_bottoms_dict, popular_bottoms, popular_tops, 
        bottom_user_dict, popular_users, ub_inter_weights_dict, tb_inter_weights_dict, ub_default_weight, tb_default_weight)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    t_len = len(test_data_ori)

    model = GPBPR(args, embedding_weight, visual_features_tensor, text_features_tensor)
    args.wide_infer = True

    if args.arch == 'VBPR':
        args.with_Nor == False
        args.with_text == False
        args.weight_P == 0
        if args.dataset == 'IQON3000':
            if args.mode == 'RB':
                args.model_path = "saved/IQON3000/VBPR_RB_79.pth" # recall@5: 0.3130, recall@10: 0.4763, ndcg@5: 0.2032, ndcg@10: 0.2557, mrr@5: 0.1672, mrr@10: 0.1887 Accuracy 0.8774 
            elif args.mode == 'RT':
                args.model_path = "saved/IQON3000/VBPR_RT_42.pth" #
        elif args.dataset == 'Polyvore_519':
            if args.mode == 'RB':
                args.model_path = "saved/Polyvore_519/VBPR/RB/VBPR_RB_46_AUC_0.6932.pth" #"saved/Polyvore_519/VBPR_RB_18.pth" #recall@5: 0.0906, recall@10: 0.1482, ndcg@5: 0.0560, ndcg@10: 0.0745, mrr@5: 0.0448, mrr@10: 0.0523 Accuracy 0.8315 AUC_NUM:  4249.0
            elif args.mode == 'RT':
                args.model_path = "saved/Polyvore_519/VBPR/RT/VBPR_RT_49_AUC_0.6980.pth" #"saved/Polyvore_519/VBPR_RT_29.pth" #recall@5: 0.0793, recall@10: 0.1564, ndcg@5: 0.0480, ndcg@10: 0.0728, mrr@5: 0.0378, mrr@10: 0.0479 Test: [40/40] Accuracy 0.7119 AUC_NUM:  3638.0
    elif args.arch == 'GPBPR':
        args.with_Nor == False
        if args.dataset == 'IQON3000':
            if args.mode == 'RB':
                args.model_path = "saved/IQON3000/GPBPR_RB_47.pth" #recall@5: 0.2913, recall@10: 0.4299, ndcg@5: 0.1871, ndcg@10: 0.2318, mrr@5: 0.1530, mrr@10: 0.1714 Accuracy 0.8566 AUC_NUM:  19783.0
            elif args.mode == 'RT':
                args.model_path = "saved/IQON3000/GPBPR_RT_22.pth" # recall@5: 0.1685, recall@10: 0.2504, ndcg@5: 0.1185, ndcg@10: 0.1448, mrr@5: 0.1022, mrr@10: 0.1129 Accuracy 0.7754 AUC_NUM:  17909.0
        elif args.dataset == 'Polyvore_519':
            if args.mode == 'RB':    
                args.model_path = "saved/Polyvore_519/GPBPR/RB/GPBPR_RB_45_AUC_0.7448.pth" #"saved/Polyvore_519/GPBPR_RB_21.pth" #recall@5: 0.1026, recall@10: 0.1711, ndcg@5: 0.0642, ndcg@10: 0.0861, mrr@5: 0.0517, mrr@10: 0.0606
            elif args.mode == 'RT':
                args.model_path = "saved/Polyvore_519/GPBPR/RT/GPBPR_RT_41_AUC_0.7397.pth" #"saved/Polyvore_519/GPBPR_RT_20.pth" #recall@5: 0.1160, recall@10: 0.1967, ndcg@5: 0.0704, ndcg@10: 0.0961, mrr@5: 0.0555, mrr@10: 0.0660 Accuracy 0.7405 AUC_NUM:  3784.0


    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    logger.info(args)
    logger.info(model)
    logger.info(args.test_data)
    model.to(args.device)
    
    metrics_w, preds_w, item_ids  = WIDE_INFER(args.device, model, test_loader, t_len=100)
    df = pd.DataFrame(item_ids.cpu().numpy())
    csv_file_path = "saved/{}/{}/{}/_infer_top100_0701.csv".format(args.dataset, args.arch, args.mode)  
    df.to_csv(csv_file_path, index=False, header=False)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    # CUDA_LAUNCH_BLOCKING=1
    main()
