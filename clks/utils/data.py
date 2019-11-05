# coding: utf-8
"""
    By cjliux@gmail.com at 2018-08-21 18:36:44
    ref: PytorchZeroToAll.git
"""
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def pad_sequences(vectorized_seqs):
    """BS"""
    seq_lengths = torch.LongTensor([len(seq) for seq in vectorized_seqs])

    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    seq_mask = torch.zeros((len(vectorized_seqs), seq_lengths.max())).float() 
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        seq_mask[idx, :seq_len].fill_(1.)

    return seq_tensor, seq_lengths, seq_mask


def pad_sequences_2d(vectorized_seqs):
    return pad_sequences(vectorized_seqs)


def pad_sequences_3d(vectorized_seqs):
    """
        vectorized_seqs: list(batch).list(group).list(len)
    BIS
    """
    n_samples = len(vectorized_seqs)
    n_insts = [len(sample) for sample in vectorized_seqs]
    len_seqs = [[len(inst) for inst in sample] for sample in vectorized_seqs]

    max_n_insts = max(n_insts)
    max_len_seqs = max(max(lseq) for lseq in len_seqs)

    seq_lengths = torch.zeros((n_samples, max_n_insts)).long()
    seq_tensor = torch.zeros((n_samples, max_n_insts, max_len_seqs)).long()
    seq_mask = torch.zeros((n_samples, max_n_insts, max_len_seqs)).float()
    
    for i_sample, (sample, n_inst) in enumerate(zip(vectorized_seqs, n_insts)):
        seq_lengths[i_sample, :n_inst] = torch.LongTensor(len_seqs[i_sample])
        for i_inst, (inst, lseq) in enumerate(zip(sample, len_seqs[i_sample])):
            seq_tensor[i_sample, i_inst, :lseq] = torch.LongTensor(inst)
            seq_mask[i_sample, i_inst, :lseq].fill_(1.)

    return seq_tensor, seq_lengths, seq_mask


def crop_sequence(sequence, max_len, training=True):
    start = 0
    if training:
        start = random.randint(0, max(len(sequence) - max_len, 0))
    return sequence[start: start+max_len]


def DBC2SBC(ustring):
    """ 全角转半角 """
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0021 <= inside_code and inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return rstring


def SBC2DBC(ustring):
    """ 半角转全角 """
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x0020:
            inside_code = 0x3000
        else:
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            inside_code += 0xfee0
        rstring += chr(inside_code)
    return rstring

