#coding: utf-8
import os
import sys

from .lr_scheduler import LRScheduler


class KeepConstant(LRScheduler):
    
    def step_epoch(self, epoch):
        pass

    def step_update(self, num_updates):
        pass
