#coding: utf-8
import torch 
import torch.nn as nn
import torch.nn.functional as F


def InfoEnt(p_dist, reduce=None):
    """
        ient = - \sum_i p(i) \log p(i)
    """
    ent = - torch.mul(p_dist, torch.log(p_dist))
    ent = torch.sum(ent, -1)
    ent = ent if reduce is None else reduce(ent)
    return ent
        

def CrossEnt(p_dist, q_dist, reduce=None):
    """
        xent = - \sum_i p(i) \log q(i)
    """
    ent = - torch.mul(p_dist, torch.log(q_dist))
    ent = torch.sum(ent, -1)
    ent = ent if reduce is None else reduce(ent)
    return ent


def KLDivergence(p_dist, q_dist, reduce=None):
    """
        KLD(p||q) = - \sum_i p(i) \log (q(i)/p(i))
    Args:
        p_dist/q_dist: Tensor
    """
    div = CrossEnt(p_dist, q_dist) - InfoEnt(p_dist, p_dist)
    div = torch.sum(div, -1)
    div = div if reduce is None else reduce(div)
    return div

