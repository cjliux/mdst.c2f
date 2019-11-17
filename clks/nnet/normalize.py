# coding: utf-8
"""
    By cjliux@gmail.com
Notes:
    ref: https://github.com/seba-1511/lstms.pth/blob/master/lstms/normalize.py
"""
import os
import sys
import copy
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F


"""
Implementation of various normalization techniques. Also only works on instances
where batch size = 1.
"""

class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf    
    """

    def __init__(self, dim_input, learnable, epsilon=1e-6):
        """
        Args:
            config:
                dim_input: int
                learnable: T or F
                epsilon: float, default 1e-6.
        """
        super().__init__()
        self.dim_input = dim_input
        self.learnable = learnable
        self.epsilon = epsilon

        self.scale = nn.Parameter(torch.ones(dim_input))
        self.offset = nn.Parameter(torch.zeros(dim_input))
        
        if not learnable:
            for p in self.parameters():
                p.requires_grad = False

        self.init_weights()

    def init_weights(self):
        # std = 1.0 / math.sqrt(self.config["dim_input"])
        # for w in self.parameters():
        #     w.data.uniform_(-std, std)
        pass

    def forward(self, tensor):
        """
        Args:
            input: (..., dim)
        """
        mu = tensor.mean(-1, keepdim=True) 
        var = (tensor - mu).pow(2).mean(-1, keepdim=True)
        u = (tensor - mu) / torch.sqrt(var + self.epsilon)
        tensor =  self.scale * u + self.offset
        return tensor


def layer_norm(tensor, scale=None, offset=None, epsilon=1e-8):
    """
        tensor: T(..., dim)
    """
    mu = tensor.mean(-1, keepdim=True)
    var = (tensor - mu).pow(2).mean(-1, keepdim=True)
    u = (tensor - mu) / torch.sqrt(var + epsilon)
    if scale is not None:
        u *= scale
    if offset is not None:
        u += offset
    return u

