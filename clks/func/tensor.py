#coding: utf-8
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch._six import inf


def total_norm(tensors, norm_type=2):
    norm_type = float(norm_type)
    if norm_type == inf:
        tot_norm = max(t.abs().max() for t in tensors)
    else:
        tot_norm = 0
        for t in tensors:
            param_norm = t.norm(norm_type)
            tot_norm += param_norm.item() ** norm_type
        tot_norm = tot_norm ** (1. / norm_type)
    return tot_norm


def to_cuda(tensor, device=None):
    if torch.cuda.is_available():
        if device is None:
            return tensor.cuda()
        else:
            return tensor.cuda(device)
            

def flip(tensor, dim=0):
    """flip tensor along one dimension"""
    inv_idx = torch.arange(tensor.size(dim)-1, -1, -1).long()
    inv_tensor = tensor.index_select(dim, to_cuda(inv_idx))
    return inv_tensor


def outer(tensor1, tensor2):
    """
    Args:
        tensor1: ([..,] dim1)
        tensor2: ([..,] dim2)
    """
    outer = torch.matmul(tensor1.unsqueeze(-1), tensor2.unsqueeze(-2))
    return outer


def instance_propagate(annots_inst, masks_inst, is_sequence=True):
    """
        propagate the last valid instance to the last position.
    Args:
        annots_inst: (inst, [seq,] ..., dim)
        masks_inst: (inst, [seq,] ...)
    Returns
        new_annots, new_masks
    Notes: 
        inst-first enforced
        if ndim > 3, then dim(1)=seq is assumed.
        useful when multiple instances are available for each sample.
    """
    if is_sequence:
        has_insts = (masks_inst.sum(1) > 0).float()
        has_insts = has_insts.unsqueeze(1).expand_as(masks_inst)
    else:
        has_insts = masks_inst

    new_annots, new_masks = [annots_inst[0]], [masks_inst[0]]
    for annot_inst, mask_inst, has_inst in zip(
            annots_inst[1:], masks_inst[1:], has_insts[1:]):
        new_mask = has_inst * mask_inst + (1 - has_inst) * new_masks[-1]
        new_masks.append(new_mask)

        has_inst = has_inst.unsqueeze(-1)
        new_annot = has_inst * annot_inst + (1 - has_inst) * new_annots[-1]
        new_annots.append(new_annot)
    new_annots = torch.stack(new_annots)
    new_masks = torch.stack(new_masks)
    return new_annots, new_masks


def instance_one_step_propagate(annots_inst, masks_inst, is_sequence=True):
    """
        propagate the last valid instance to the last position.
    Args:
        annots_inst: (inst, [seq,] ..., dim)
        masks_inst: (inst, [seq,] ...)
    Returns
        new_annots, new_masks: seq = seq + 1 (seq len inc by 1)
    Notes: 
        inst-first enforced
        if ndim > 3, then dim(1)=seq is assumed.
        useful when multiple instances are available for each sample.
    """
    if is_sequence:
        has_insts = (masks_inst.sum(1) > 0).float()
        has_insts = has_insts.unsqueeze(1).expand_as(masks_inst)
    else:
        has_insts = masks_inst

    n_insts = annots_inst.size(0)
    annots_inst = torch.cat([annots_inst[0].unsqueeze(0), annots_inst, 
        to_cuda(torch.zeros_like(annots_inst[0].unsqueeze(0)))])
    masks_inst = torch.cat([masks_inst[0].unsqueeze(0), masks_inst, 
        to_cuda(torch.zeros_like(masks_inst[0].unsqueeze(0)))])
    inv_annots_inst, inv_masks_inst = flip(annots_inst), flip(masks_inst)
    inv_has_insts = flip(has_insts)

    new_annots, new_masks = [], []
    for annot_inst, mask_inst, has_inst, prev_annot_inst, prev_mask_inst in zip(
            inv_annots_inst[:-1], inv_masks_inst[:-1], inv_has_insts[:-1],
            inv_annots_inst[1:], inv_masks_inst[1:]):
        new_mask = has_inst * mask_inst + (1 - has_inst) * prev_mask_inst
        new_masks.append(new_mask)

        has_inst = has_inst.unsqueeze(-1)
        new_annot = has_inst * annot_inst + (1 - has_inst) * prev_annot_inst
        new_annots.append(new_annot)
    new_annots = torch.stack(new_annots[::-1])
    new_masks = torch.stack(new_masks[::-1])
    return new_annots, new_masks

