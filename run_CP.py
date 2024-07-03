import argparse
import os
import pdb
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csr_matrix
from torch.optim import Adam
from trainer.utility import Dataset
from Models.graph.TransMatch_ori import TransMatch
from tool.metrics import *


import time
import logging
import torch.nn.functional as F
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

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def evaluating(model, testData, device, topks):
    model.eval()
    preds = []
    for iteration, aBatch in enumerate(testData):
        aBatch = [x.to(device) for x in aBatch]
        scores = model.inference(aBatch).detach().cpu()
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
    # conf['cate_num'] = len(dataset.cate_items)
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
    model.to(conf['device'])
    early_stopping = EarlyStopping(patience=conf['patience'], verbose=True)

    optimizer = Adam([{
        'params': model.parameters(),
        'lr': conf['lr'],
        'weight_decay': conf['wd']
    }])
    performance_files, result_path, model_path = get_save_file(
        conf, dataset.test_setting_list)
    model_name = '_'.join(result_path.split('/')[-3:-1])
    if conf['save_results'] or conf['save_model']:
        best_auc = 0
        best_hit = 0

    for epoch in range(conf['max_epoch']):
        model.train()
        loss_scalar = 0.
        savename = None

        for iteration, aBatch in enumerate(dataset.train_loader):
            aBatch = [x.to(conf['device']) for x in aBatch]
            loss = model.forward(aBatch)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_scalar += loss.detach().cpu()
        curr_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('%s Epoch %d Loss: %.6f' %
              (curr_time, epoch, loss_scalar / iteration))
        test_results = {}
        if epoch % conf['evaluation_interval'] == 0:
            print(model_name)
            for test_setting, test_loader in zip(dataset.test_setting_list,
                                                 dataset.test_loader_list):
                if 'auc' in test_setting:  # auc evalution
                    metrics, preds = evaluating(model, test_loader,
                                                conf['device'], conf['topk'])
                    curr_time = '%s ' % datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')
                    epoch_str = 'Epoch %d' % epoch
                    result_str = ''
                    for met in metrics[1]:
                        auc = metrics[1][met]
                        result_str += ' {}: {:.4f}'.format('AUC', auc)
                        break

                    if conf['save_model']:
                        # if (epoch % conf['save_freq'] == 0):
                            # savename = model_path + 'epoch_%d_%s_AUC_%.4f_mode_%s.pth' %(epoch, test_setting, best_auc, conf['mode'])
                            
                        if auc > best_auc:
                            best_auc = auc
                            savename = './saved/' + conf['dataset'] + '/'+ conf['model'] + '/p%dc%d_%s_AUC_.pth' %(conf['path'], conf['context'],conf['mode'])
                            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, savename)
                            # shutil.rmtree(model_path)
                            # os.makedirs(model_path)
                        # elif auc < best_auc:
                            # if epoch / conf['save_freq'] > 2:
                            #     deletename =  './saved/' + conf['dataset'] + '/'+ conf['model'] + '/epoch_%d_p%dc%d_%s.pth' %(epoch - conf['save_freq']*2,conf['path'], conf['context'], conf['mode'])     
                            #     os.remove(deletename)

                    print('%s' % test_setting[:-5], curr_time, result_str)
                    output_f = open(performance_files[test_setting], 'a')
                    output_f.write(curr_time + 'Epoch %d' % epoch +
                                   result_str + '\n')
                    output_f.close()

                else:  # topk evaluation
                    metrics, preds = evaluating(model, test_loader,
                                                conf['device'], 
                                                conf['topk'])
                    curr_time = '%s ' % datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')
                    epoch_str = 'Epoch %d' % epoch
                    output_f = open(performance_files[test_setting], 'a')
                    result_str = ''
                    for topk in conf['topk']:
                        for met in metrics[conf['topk'][0]]:
                            hit = metrics[topk][met]
                            result_str += ' %s@%d: %.4f' % ('Hit', topk, hit)
                            break

                    print('%s' % (test_setting[:-5]), curr_time, result_str)
                    output_f.write(curr_time + 'Epoch %d' % epoch +result_str + '\n')
                    output_f.close()

        early_stopping(auc, model)
        if early_stopping.early_stop:
            print('Early stopping')
            if savename is not None and os.path.exists(savename):
                new_name_for_savename = savename.replace('.pth', '_AUC_%.4f.pth'%auc)
                os.rename(savename, new_name_for_savename)
            break


def get_save_file(conf, settings):
    if conf['model'] == 'TransMatch':
        f_name = 'TransMatch_' + conf['score_type']
        if conf['context']:
            f_name += '_pcc'
            f_name += '_' + str(conf['context_hops'])
            f_name += '_' + str(conf['neighbor_samples'])
            f_name += '_%s' % conf['neighbor_agg']
            f_name += '_%.2f' % conf['agg_param']
        if conf['path']:
            f_name += '_pcp_%d_%d_%.2f_%s' % (
                conf['path_num'], conf['max_path_len'], conf['path_weight'],
                conf['path_agg'])

    f_name += '/'
    performance_path = conf['root_path'] + conf['performance_path'] + f_name
    model_path = conf['root_path'] + conf['model_path'] + f_name
    result_path = conf['root_path'] + conf['result_path'] + f_name
    f_list = {}
    if not os.path.exists(performance_path):
        os.makedirs(performance_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for setting in settings:
        output_file = performance_path + setting
        f_list[setting] = output_file
    return f_list, result_path, model_path


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
    print(conf)
    conf['device'] = torch.device(
        'cuda:%s' % conf['gpu'] if torch.cuda.is_available() else 'cpu')

    if conf['wide_evaluate']:
        conf['test_batch_size'] = 64

    conf['performance_path'] += (conf['dataset'] + '/')
    conf['result_path'] += (conf['dataset'] + '/')
    conf['model_path'] += (conf['dataset'] + '/')
    print(conf)
    Train_Eval(conf)
