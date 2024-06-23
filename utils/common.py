import copy
import warnings
import numpy as np
import math
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.optim.lr_scheduler import _LRScheduler

# def adjust_learning_rate(optimizer, epoch, lr, batch_size):
#     """
#     Decay the learning rate with half-cycle cosine after warmup
#     warmup_epochs : 40; min_lr : 0
#     """
#     warmup_epochs = 10
#     # min_lr = 1e-6
#     min_lr = 0
#     # total_epochs = 500
#     total_epochs = batch_size
#     # init_lr = 1e-3
#     init_lr = lr

#     if epoch < warmup_epochs:
#         lr = init_lr * (epoch / warmup_epochs)
#     else:
#         lr = min_lr + (init_lr - min_lr) * 0.5 * \
#             (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
#     for param_group in optimizer.param_groups:
#         if "lr_scale" in param_group:
#             param_group["lr"] = lr * param_group["lr_scale"]
#         else:
#             param_group["lr"] = lr
#     return lr

class adjust_lr(_LRScheduler):
    def __init__(self, optimizer, init_lr = 1e-3, epochs = 300, warmup_epochs = 10, min_lr = 1e-6, decay_rate = 0.9, decay_step = 20,):
        self.init_lr = init_lr
        self.total_epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # lr = self.init_lr * (self.last_epoch / self.warmup_epochs)
            lr = self.min_lr + (self.init_lr-self.min_lr) * (self.last_epoch / self.warmup_epochs)
        else:
            lr = self.min_lr + (self.init_lr - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
            # lr = self.min_lr + (self.init_lr - self.min_lr) * math.exp(-self.decay_rate*(self.last_epoch - self.warmup_epochs))   # not valid
            # lr = self.min_lr + (self.init_lr - self.min_lr) * self.decay_rate**((self.last_epoch - self.warmup_epochs)/self.decay_step)   # ok
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return [group['lr'] for group in self.optimizer.param_groups]

#customize exp lr scheduler with min lr
class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma    
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)    
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


def repeat_data(data: Data, num_repeat) -> Batch:
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    return Batch.from_data_list(datas)

def repeat_batch(batch: Batch, num_repeat) -> Batch:
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
    elif cfg.type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
        )
    elif cfg.type == 'expmin':
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=cfg.factor,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'expmin_milestone':
        gamma = np.exp(np.log(cfg.factor) / cfg.milestone)
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=gamma,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'defined':
        return adjust_lr(optimizer, init_lr=cfg.init_lr, epochs=cfg.max_iters, warmup_epochs=cfg.warmups, min_lr=cfg.min_lr)
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)