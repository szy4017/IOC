import os
import argparse
import logging
import random
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from dataloader.ocl import OCLDataset
from dataloader.rtf import RTFDataset


def main(configs):

    # Load data
    # dataset = load_dataset(dataset_name, data_path, normal_class)
    if configs.dataset == 'OCL':
        train_set = OCLDataset(configs.scene, type = 'train')
        test_set = OCLDataset(configs.scene, type = 'test')
    elif configs.dataset == 'RTF':
        train_set = RTFDataset(type = 'train')
        test_set = RTFDataset(type = 'test')
    logger.info(f'Train_set has {len(train_set)} pictures; Test_set has {len(test_set)} pictures')
    train_loader = data.DataLoader(train_set, batch_size= configs.batch_size)
    test_loader = data.DataLoader(test_set, batch_size = configs.batch_size)

    # Load model
    model = base_Model(configs, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay,
                               amsgrad=configs.optimizer == 'amsgrad')
    # Set learning rate scheduler

    if not configs.test:
        indices, labels, scores = Trainer(model, optimizer, train_loader, test_loader, test_loader, device, logger, configs, xp_path)

    else:
        # model = torch.load(os.path.join(xp_path,  str(configs.scene).zfill(2) + '_best_network.pkl'))
        indices, labels, scores = Tester(test_loader, device, logger, configs, xp_path)


    # idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score
    idx_sorted = indices[np.argsort(scores)]

    X_normals = test_set.test_data[idx_sorted[:10]]
    X_outliers = test_set.test_data[idx_sorted[-10:]]

    plot_images_grid(X_normals, export_img=xp_path + '/normals', title='Most normal examples', padding=2)
    plot_images_grid(X_outliers, export_img=xp_path + '/outliers', title='Most anomalous examples', padding=2)


if __name__ == '__main__':

    '''Experiment Args'''

    parser = argparse.ArgumentParser()

    home_dir = os.getcwd()
    parser.add_argument('--dataset', default='OCL', type=str,
                        help='Dataset of choice: OCL, RTF')
    parser.add_argument('--scene', default='1', type=str, 
                        help='[1,20]')
    parser.add_argument('--method', default='svae', type=str,
                        help='Model: svdd, svae, innd')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Computation device')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed value')
    parser.add_argument('--test', default=False, type=bool,
                        help='train or test')
    parser.add_argument('--pretrain', default=False, type=bool,
                        help='Pretrain neural network parameters via autoencoder.')
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='[adam, amsgrad]')
    parser.add_argument('--n_jobs_dataloader', type=int, default=0,
                help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                        help='saving directory')

    args = parser.parse_args()

    xp_path = f'result/{args.dataset}/{args.method}'
    if not os.path.exists(xp_path):
        os.makedirs(xp_path)

    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    experiment_log_dir = os.path.join(args.logs_save_dir, args.dataset, args.method, f"_seed_{args.seed}")
    os.makedirs(experiment_log_dir, exist_ok=True)
    log_file = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    exec(f'from conf.{args.dataset}.{args.method}_Configs import Config as Configs')
    exec(f'from models.{args.method}.{args.method}_network.model import base_Model')
    exec(f'from models.{args.method}.{args.method}_trainer.trainer import Trainer')
    exec(f'from models.{args.method}.{args.method}_trainer.trainer import Tester')


    configs = Configs()
    configs.__dict__.update(vars(args))
    
    
    main(configs)
