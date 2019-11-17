import torch.nn as nn
from abc import ABCMeta, abstractmethod


class Model(nn.Module):

    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()

    @abstractmethod
    def init_weights(self):
        pass

    # @abstractmethod
    # def forward(self, *args, **kwargs):
    #     pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def inference(self, *args, **kwargs):
        pass

    @abstractmethod
    def trainable_parameters(self):
        pass

    def before_update(self, global_count):
        pass

    def before_epoch(self, curr_epoch):
        pass

