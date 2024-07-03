import os
import yaml
import argparse
from util import config
import torch

def parse_configure():
    parser = argparse.ArgumentParser(description='APCL')
    parser.add_argument('--arch', type=str, default='APCL', help='Model name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size number')
    parser.add_argument('--test_batch_size', type=int, default=512, help='batch_size number')
    parser.add_argument('--patience', type=int, default='10', help='patience for early stop')
    parser.add_argument('--config', type=str, default='config/APCL_Polyvore_519_RB.yaml', help='config file') #APCL_IQON3000_RB.yaml #APCL_Polyvore_RB.yaml
    parser.add_argument('opts', help='see config/APCL_Polyvore_RB.yaml for all options', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('-d',
                        '--dataset',
                        default='Polyvore_519',
                        type=str,
                        help='Polyvore_519, IQON3000, iqon_s')
    parser.add_argument('-g',
                        '--cuda',
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
                        default= 1,
                        type=int,
                        help='using context enhance branch')
    args = parser.parse_args()
    
    args.device = torch.device('cuda:%s' % args.cuda if torch.cuda.is_available() else 'cpu')
    
    # if args.model == None:
    #     raise Exception("Please provide the model name through --model.")
    # model_name = args.arch#.lower()

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    # cfg = yaml.safe_load(open(args.config))

    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
        if args.arch == 'APCL':
            cfg.config = 'config/APCL_' + str(args.dataset) + '_' + str(args.mode) + '.yaml'
        elif args.arch == 'TransMatch':
            cfg.config = 'config/CP_config.yaml'
            cfg.path = args.path
            cfg.context = args.context
            cfg.save_model = args.save_model
            cfg.path_enhance = args.path_enhance
            cfg.context_enhance = args.context_enhance
        else:
            cfg.config = 'config/'  + str(args.dataset)+ '_' + str(args.mode) + '.yaml'
        cfg.device = args.device
        cfg.arch = args.arch
        cfg.dataset = args.dataset
        cfg.mode = args.mode
        cfg.cuda = args.cuda
        cfg.batch_size = args.batch_size
        cfg.test_batch_size = args.test_batch_size
        cfg.patience = args.patience  
        # cfg.performance_path += (args.dataset + '/')
        # cfg.result_path += (args.dataset + '/')
        # cfg.model_path += (args.dataset + '/')     
    return cfg

class ConfigManager:
    def __init__(self):
        self.args = parse_configure()
        self.config_path = self.args.config
     
    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def save_config(self):
        with open(self.config_path, 'w') as file:
            yaml.safe_dump(self.config_path, file)
    
    def update_config(self, key, value):
        setattr(self.args, key, value)
    
    def ensure_path(self, key, dataset, root_datapath):
        dataset_paths = {
            'IQON3000': 'IQON3000/data/',
            'Polyvore_519': 'Polyvore_519/polyvore_U_519_data/',
            'iqon_s': 'iqon_s/',
            'ifashion': 'iFashion/'
        }
        
        if dataset in dataset_paths:
            new_datapath = os.path.join(root_datapath, dataset_paths[dataset])
            self.update_config(key, new_datapath)
            
            if not os.path.exists(new_datapath):
                os.makedirs(new_datapath)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    def get_config(self):
        return self.args
        
    

    # if not os.path.exists('./config/{}_{}_{}.yaml'.format(args.arch, args.dataset, args.mode)): #config/APCL_Polyvore_RB.yaml
    #     raise Exception("Please create the yaml file for your model first.")
    
    # # with open('./config/{}_{}_{}.yaml'.format(model_name, args.dataset, args.mode), encoding='utf-8') as f:
    # with open('./config/APCL_Polyvore_519_RB.yaml', encoding='utf-8') as f:
    #     config_data = f.read()
    #     configs = yaml.safe_load(config_data)     
    #     configs['TRAIN']['arch'] = args.arch
    #     # # grid search
    #     # if 'tune' not in configs:
    #     #     configs['tune'] = {'enable': False}

    #     configs['device'] = args.device
    #     if args.dataset is not None:
    #         configs['dataset'] = args.dataset

    #     # if 'log_loss' not in configs['train']:
    #     #     configs['train']['log_loss'] = True
    #     if args.batch_size is not None:
    #         configs['TRAIN']['batch_size'] = args.batch_size
    #         configs['TEST']['test_batch_size'] = args.test_batch_size

    #     # early stop
    #     if 'patience' in configs['TRAIN']:
    #         if configs['TRAIN']['patience'] <= 0:
    #             raise Exception("'patience' should be greater than 0.")
    #         else:
    #             configs['TRAIN']['early_stop'] = True
    #     else:
    #         configs['TRAIN']['early_stop'] = False
    #     return configs
        

# configs = parse_configure()
