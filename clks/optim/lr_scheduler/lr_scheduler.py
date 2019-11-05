#coding: utf-8
import abc
import torch
import torch.optim


class LRScheduler(object):
    """
    stateless object
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, optimizer, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        self.optimizer = optimizer

    def get_lrs(self):
        lrs = [grp['lr'] for grp in self.optimizer.param_groups]
        return lrs
    
    def set_lrs(self, lrs):
        for grp, lr in zip(self.optimizer.param_groups, lrs):
            grp['lr'] = lr

    @abc.abstractmethod
    def step_epoch(self, epoch):
        """Update the learning rate at the end of the given epoch."""
        raise NotImplementedError

    @abc.abstractmethod
    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        raise NotImplementedError
