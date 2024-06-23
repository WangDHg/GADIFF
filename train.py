import os
import time
import random
import logging
import torch
import numpy as np
from glob import glob
import math


import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter

import shutil
import argparse
import datetime
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import sys

# qt
import shutil
from argparse import ArgumentParser
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob

# from torch_geometric.data import DataLoader
# from main_run import main_process
from utils.transforms import *
from utils.misc import *
from models.network import get_model
from utils.transforms import CountNodesPerGraph
from utils.datasets import ConformationDataset
from tqdm.auto import tqdm
from torch_geometric.data import DataLoader
from utils.common import get_optimizer, get_scheduler
from torch.nn.utils import clip_grad_norm_

# from evaluate_gen import gen_rmsd

def get_logger(name, log_dir=None, log_fn='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s::%(name)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_fn))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def args_def():
    # Arguments
    parser = ArgumentParser()
    
    # start
    parser.add_argument('--config', type=str, default='configs/qm9_para.yml')
    
    # resume
    # parser.add_argument('--config', type=str, default='logs/qm9_para_2024_02_17__00_52_23')
    
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--logdir', type=str, default='logs')
    args = parser.parse_args()
    return args


def train(it, scaler_train, accum_iter, model, optimizer_model, scheduler_model, model_max_norm_value, train_loader_training, args, config_name, config, logger, writer):
    model.train()
    loss_total = []
    optimizer_model.zero_grad()

    for idx, batch in enumerate(tqdm(train_loader_training, desc='Training')):

        batch.to(args.device)
            
        if config_name[0:3] == 'nci':
            batch_edge_type = None
        else:
            batch_edge_type = batch.edge_type

        loss = model.get_loss(
            atom_type=batch.atom_type,
            pos=batch.pos,
            bond_index=batch.edge_index,
            bond_type=batch_edge_type,
            batch=batch.batch,
            num_nodes_per_graph=batch.num_nodes_per_graph,
            num_graphs=batch.num_graphs,
            anneal_power=config.train.anneal_power,
            return_unreduced_loss=True,
            extend_radius=True
        )

        loss_total.append(loss.mean().item())
        loss = loss.mean()

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)
        
        loss.backward()
        
        if (idx+1) % accum_iter == 0:
            # orig_grad_norm_model = clip_grad_norm_(model.parameters(), model_max_norm_value)  # return total_norm
            optimizer_model.step()
            optimizer_model.zero_grad()
            
            scheduler_model.step()
    
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0* 1024.0)
    
    logger.info('[Training] Iter %03d | Loss %.2f | LR %.6f | Mem(GB) %.2f ' % (
                it, torch.tensor(loss_total).mean(), optimizer_model.param_groups[0]['lr'], memory_used,
                ))

    writer.add_scalar('Training/loss', torch.tensor(loss_total).mean(), it)
    writer.add_scalar('Training/lr', optimizer_model.param_groups[0]['lr'], it)
    writer.flush()
    
    return torch.tensor(loss_total).mean()

def evaluate(it, data_loader, tag_str, model, scheduler_model, args, config_name, config, logger, writer, config_path):
    sum_loss, sum_n = 0, 0
    
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(data_loader, desc=tag_str, colour='CYAN')):
            batch.to(args.device)
            
            if config_name[0:3] == 'nci':
                batch_edge_type = None
            else:
                batch_edge_type = batch.edge_type
            
            loss = model.get_loss(
                atom_type=batch.atom_type,
                pos=batch.pos,
                bond_index=batch.edge_index,
                bond_type=batch_edge_type,
                batch=batch.batch,
                num_nodes_per_graph=batch.num_nodes_per_graph,
                num_graphs=batch.num_graphs,
                anneal_power=config.train.anneal_power,
                return_unreduced_loss=True,
                extend_radius=True
            )
            
            sum_loss += loss.sum().item()
            sum_n += loss.size(0)   # N
            
    avg_loss = sum_loss / sum_n
        
    # if config.train.model_para.scheduler.type == 'plateau':
    #     scheduler_model.step(avg_loss)
    # else:
    #     scheduler_model.step()

    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0 * 1024.0)
    
    logger.info('[%s] Iter %03d | Loss %.2f | Mem(GB) %.2f ' % (
        tag_str, it, avg_loss, memory_used,
    ))

    writer.add_scalar(tag_str+'/loss', avg_loss, it)
    writer.add_scalar(tag_str+'/mem', memory_used)
    writer.flush()
    return avg_loss


def main_process(start_time):
    
    args = args_def()
    
    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    
    used_seed(config.train.seed)
    
    # Logging
    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        shutil.copytree('models', os.path.join(log_dir, 'models'))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # QM9, Drug, NCI => Datasets and loaders
    
    logger.info('Loading datasets...')
    transforms = CountNodesPerGraph()
    train_set = ConformationDataset(config.dataset.train, transform=transforms)
    val_set = ConformationDataset(config.dataset.val, transform=transforms)
    train_loader_training = DataLoader(train_set, config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)
    
    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)

    # Optimizer
    optimizer_model = get_optimizer(config.train.model_para.optimizer, model)
    scheduler_model = get_scheduler(config.train.model_para.scheduler, optimizer_model)
    start_iter = 1


    epoch_num = []
    train_loss_list = []
    val_loss_list = []
    model_max_norm_value = config.train.model_max_grad_norm

    # Resume from checkpoint
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
        loss_path = os.path.join(os.path.join(resume_from, 'checkpoints'), '%d.pkl' % start_iter)
        logger.info('Resuming from: %s' % ckpt_path)
        logger.info('Iteration: %d' % start_iter)
        start_iter += 1
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer_model.load_state_dict(ckpt['optimizer_model'])
        scheduler_model.load_state_dict(ckpt['scheduler_model'])
        
        loss_para = torch.load(loss_path)
        epoch_num.append(loss_para['epoch_num'])
        train_loss_list.append(loss_para['train_loss_list'])
        val_loss_list.append(loss_para['val_loss_list'])
    
    scaler_train = GradScaler()

    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train_loss = train(it, scaler_train, config.train.accum_iter, model, optimizer_model, scheduler_model, model_max_norm_value, train_loader_training, args, config_name, config, logger, writer)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = evaluate(it, val_loader, "Validate", model, scheduler_model, args, config_name, config, logger, writer, config_path)
                
                epoch_num.append(it)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                 
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer_model': optimizer_model.state_dict(),
                    'scheduler_model': scheduler_model.state_dict(),
                    'val_loss': val_loss,
                }, ckpt_path)
                
                loss_path = os.path.join(ckpt_dir, '%d.pkl' % it)
                torch.save({
                    'epoch_num': epoch_num,
                    'train_loss_list': train_loss_list,
                    'val_loss_list': val_loss_list,
                }, loss_path)
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))
        
    except KeyboardInterrupt:
        logger.info('Terminating...')

    
if __name__ == "__main__":
    start_time = time.time()
    main_process(start_time)
    
    

    