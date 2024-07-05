import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import yaml
from trainer.utility import Dataset
from Models.graph.TransMatch_ori import TransMatch
from tool.metrics import *

import time
import logging
from util import config
from tool.util import * #AverageMeter, poly_learning_rate, find_free_port, EarlyStopping
import csv
from sys import argv
import json
from torch.nn import *
import random
from collections import defaultdict
from tqdm import tqdm
from config.configurator import parse_configure

##注意如果使用多gpu,把TransMatch_ori inference 改成forward
## CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python wide_infer_CP.py -d=IQON3000 -p=1 -c=1 -s=0 -m=RB

def WIDE_INFER(model, testData, device, topks): #for wide infer
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()
    end = time.time()
    pos = 0
    preds = []
    top_scores = []
    ordered_item_ids = []
    with torch.no_grad(): 
        for iteration, aBatch in enumerate(testData):
            
            aBatch = [x.to(device) for x in aBatch]
            # itemid= torch.cat([aBatch[2].unsqueeze(1), aBatch[3].expand(-1, args.test_batch_size)], dim=-1)
            itemid= torch.cat([aBatch[2].unsqueeze(1), aBatch[3]], dim=-1)
            # scores = model.inference(aBatch, train=False)      
            # pos += float(torch.sum(output.ge(0)))
            # scores = model.inference(aBatch)#.detach().cpu()
            scores = model(aBatch)#!!!把模型的inference和forward互换！！
            # print(scores.size()) #torch.Size([2048, 21])
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
    for topk in topks: #args.k:
        metrics[topk] = {}
        REC, MRR, NDCG = get_metrics(grd, grd_cnt, preds.cpu().numpy(), topk)
        metrics[topk]['recall'] = REC
        metrics[topk]['mrr'] = MRR
        metrics[topk]['ndcg'] = NDCG
    # AUC = pos/t_len
    batch_time.update(time.time() - end)
    end = time.time()
    metric_strings = []
    for m in metrics[topk]: #hit,  mrr, ndcg
        for k in topks: #[5,10,20]
            metric_strings.append('{}@{}: {:.4f}'.format(m, k, metrics[k][m]))
    print(', '.join(metric_strings))
    return metrics, preds, ordered_item_ids[:, :topks[-1]]  #, top_scores

def evaluating(model, testData, device, topks):
    model.eval()
    preds = []
    for iteration, aBatch in enumerate(testData):
        aBatch = [x.to(device) for x in aBatch]
        scores = model.inference(aBatch).detach().cpu()
        # print(scores.size()) #torch.Size([2048, 2])
        _, tops = torch.topk(scores, k=topks[-1], dim=-1) #k=topks[-1]
        preds.append(tops)

    preds = torch.cat(preds, dim=0)
    bs = preds.size(0)
    grd = [0] * bs
    grd_cnt = [1] * bs
    metrics = {}
    for topk in topks:
        metrics[topk] = {}
        REC, MRR, NDCG = get_metrics(grd, grd_cnt, preds.numpy(), topk)
        metrics[topk]['recall'] = REC
        metrics[topk]['mrr'] = MRR
        metrics[topk]['ndcg'] = NDCG
    return metrics, preds

def continue_training(model_path):
    model = torch.load(model_path)
    print('Continuing training with existing model...')


def Train_Eval(conf):
    dataset = Dataset(conf)
    conf['user_num'] = len(dataset.user_map)
    conf['item_num'] = len(dataset.item_map)
    if conf['dataset'] == 'iqon_s':
        conf['cate_num'] = len(dataset.cate_items)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(
        'data prepared, %d users, %d items, %d train, %d test, %d validation data'
        % (len(dataset.user_map), len(dataset.item_map), len(
            dataset.traindata), len(dataset.testdata), len(dataset.valdata)))
    if conf['model'] == 'TransMatch':
        if conf['pretrain_mode']:
            #     pretrain_model_file = f"{conf['model']}-{'iqon_s'}-{'pretrained_model'}.pth.tar"
            #     pretrain_model_path = "model/iqon_s/pretrained_model/" + pretrain_model_file
            #     if os.path.exists(pretrain_model_path):
            #         model = torch.load(pretrain_model_path)
            #         print("Continuing training with existing model...")
            #     else:
            #         model = BPR(conf, conf["user_num"], conf["item_num"], conf['hidden_dim'], conf['score_type'], dataset.visual_features.to(conf["device"]))
            # else:
            # train_data_path = conf["root_path"] + "/data/iqon_s/train.csv"
            # adj_UJ, adj_IJ, all_top_ids, all_bottom_ids, all_users_ids, top_idx_to_encoded, bottom_idx_to_encoded = build_adj(train_data_path)
            model = TransMatch(
                conf, dataset.neighbor_params,
                dataset.visual_features.to(conf['device'])
            )  #, adj_UJ.to(conf["device"]), adj_IJ.to(conf["device"]), all_top_ids, all_bottom_ids, all_users_ids, top_idx_to_encoded, bottom_idx_to_encoded)
        else:
            model = TransMatch(conf, dataset.neighbor_params,
                               dataset.visual_features.to(conf['device']))
    if conf['dataset'] == "Polyvore_519":
        if conf["mode"] == "RB":
            model_path = "saved/Polyvore_519/TransMatch/epoch_142_p1c1_RB_AUC_0.8115.pth"
        elif conf["mode"] == "RT":
            model_path = "saved/Polyvore_519/TransMatch/p1c1_RT_AUC_.pth"
    elif conf['dataset'] == "IQON3000":
        if conf["mode"] == "RB":
            model_path =  "saved/IQON3000/TransMatch/p1c1_RB_AUC_.pth"
        elif conf["mode"] == "RT":
            model_path = "saved/IQON3000/TransMatch/epoch_121_p1c1_RT_AUC_0.8523.pth"
    
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}'".format(model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))
    
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model.to(conf['device'])
   
    for test_setting, test_loader in zip(dataset.test_setting_list,
                                            dataset.test_loader_list):
        # if 'auc' in test_setting:  # auc evalution
        #     metrics, preds = evaluating(model, test_loader,
        #                                 conf['device'], conf['topk'])
        #     curr_time = '%s ' % datetime.now().strftime(
        #         '%Y-%m-%d %H:%M:%S')
        #     result_str = ''
        #     for met in metrics[1]:
        #         auc = metrics[1][met]
        #         result_str += ' {}: {:.4f}'.format('AUC', auc)
        #         break
        #     print('%s' % test_setting[:-5], curr_time, result_str)
        
        # else:  # topk evaluation
            metrics, preds, item_ids = WIDE_INFER(model, test_loader,
                                        conf['device'], 
                                        conf['topk'])
            curr_time = '%s ' % datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') 
            result_str = ''
            for topk in conf['topk']:
                for met in metrics[conf['topk'][0]]:
                    hit = metrics[topk][met]
                    result_str += ' %s@%d: %.4f' % ('Hit', topk, hit)
                    break
            print('%s' % (test_setting[:-5]), curr_time, result_str)
            df = pd.DataFrame(item_ids.cpu().numpy())
            csv_file_path = "saved/{}/{}/{}/_infer_top100_0703.csv".format(conf['dataset'], conf['model'], conf['mode'])  
            df.to_csv(csv_file_path, index=False, header=False)      

def get_cmd():
    parser = argparse.ArgumentParser()
    # general params
    parser.add_argument('-d',
                        '--dataset',
                        default='Polyvore_519',
                        type=str,
                        help='Polyvore_519, IQON3000, iqon_s')
    parser.add_argument('-g',
                        '--gpu',
                        default='0',
                        type=str,
                        help='assign cuda device')
    parser.add_argument('-p',
                        '--path',
                        default= 1,
                        type=int,
                        help='using path branch')
    parser.add_argument('-c',
                        '--context',
                        default= 1,
                        type=int,
                        help='using context branch')
    parser.add_argument('-s',
                        '--save_model',
                        default= 1,
                        type=int,
                        help='save trained model')
    parser.add_argument('-m',
                        '--mode',
                        default= "RB",
                        type=str,
                        help='given top recommend bottom')
    parser.add_argument('-PE',
                        '--path_enhance',
                        default= 0,
                        type=int,
                        help='using path enhance branch')
    parser.add_argument('-CE',
                        '--context_enhance',
                        default= 0,
                        type=int,
                        help='using context enhance branch')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    paras = get_cmd().__dict__
    conf = yaml.safe_load(open('./config/CP_config.yaml'))
    for k in paras:
        conf[k] = paras[k]
    conf['device'] = torch.device(
        'cuda:%s' % conf['gpu'] if torch.cuda.is_available() else 'cpu')

    conf['wide_evaluate']=True
    conf['test_batch_size'] =1

    conf['performance_path'] += (conf['dataset'] + '/')
    conf['result_path'] += (conf['dataset'] + '/')
    conf['model_path'] += (conf['dataset'] + '/')
    conf['wide_evaluate'] = True
    conf['topk'] = [5,10,20, 100]
    conf['neg_num'] = 99
    # conf['mode'] = "RT"
    print(conf)
    Train_Eval(conf)
