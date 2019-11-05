#coding: utf-8
import os
import sys

import torch
import torch.optim
from .lr_scheduler import LRScheduler


class InverseLinearDecayByEpoch(LRScheduler):

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.init_lrs = self.get_lrs()

    def step_epoch(self, epoch):
        self.set_lrs([lr / (1. + self.decay_rate * epoch) for lr in self.init_lrs])
    
    def step_update(self, num_updates):
        pass

