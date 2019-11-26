#coding: utf-8
import os
import math
import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import clks.func.tensor as T


class GRU(nn.Module):

    def __init__(self, dim_input, dim_hidden, bias=True,
            n_layers=1, bidirectional=False,
            drop_method='none', drop_prob=0, 
            batch_first=False, reverse=False):
        super().__init__()
        self.dim_input, self.dim_hidden = dim_input, dim_hidden
        self.n_layers, self.bidirectional = n_layers, bidirectional
        self.batch_first = batch_first

        self.layers = []
        input_size = dim_input
        for i_layer in range(n_layers):
            self.layers.append(GRUCell(input_size, dim_hidden, bias, 
                drop_method, drop_prob, batch_first=False))
            if bidirectional:
                self.layers.append(GRUCell(input_size, dim_hidden, bias, 
                    drop_method, drop_prob, batch_first=False, reverse=True))
                input_size = 2 * dim_hidden
            else:
                input_size = dim_hidden
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input, mask=None, state=None):
        if mask is None:
            mask = torch.ones(*input.size()[:-1])
        if self.batch_first:
            input, mask = input.transpose(0,1), mask.transpose(0,1)
        batch_size = input.size(1)
        n_cells = 2 * self.n_layers if self.bidirectional else self.n_layers
        if state is None:
            state = T.to_cuda(torch.zeros(n_cells, batch_size, self.dim_hidden))
        i_layer = 0
        output = input
        new_states = []
        while i_layer < n_cells:
            state_i = state[i_layer].unsqueeze(-1)
            output_fw, state_i = self.layers[i_layer](output, mask, state_i)
            new_states.append(state_i)
            i_layer += 1

            if self.bidirectional:
                state_i = state[i_layer].unsqueeze(-1)
                output_bw, state_i = self.layers[i_layer](output, mask, state_i)
                new_states.append(state_i)
                i_layer += 1
                output = torch.cat((output_fw, output_bw), -1)
            else:
                output = output_fw
        state = torch.cat(state)

        if self.batch_first:
            output = output.transpose(0,1)
        return output 


class GRUCell(nn.Module):

    def __init__(self, dim_input, dim_hidden, bias=True,
            drop_method='none', drop_prob=0, 
            batch_first=False, reverse=False):
        super().__init__()
        self.dim_input, self.dim_hidden = dim_input, dim_hidden
        self.drop_method, self.drop_prob = drop_method, drop_prob
        self.batch_first = batch_first
        self.reverse = reverse

        self.x2zr = nn.Linear(dim_input, 2*dim_hidden, bias)
        self.h2zr = nn.Linear(dim_hidden, 2*dim_hidden, bias=False)
        self.x2h = nn.Linear(dim_input, dim_hidden, bias)
        self.h2h = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.dim_hidden)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def __drop_scaled(self, value):
        keep_prob = 1.0 - self.drop_prob
        self.mask = T.to_cuda(torch.bernoulli(
            torch.Tensor(1, self.dim_hidden).fill_(keep_prob)))
        value.data.set_(torch.mul(value, self.mask).data)
        value.data *= 1.0/(1.0 - self.drop_prob)

    def forward(self, input, mask=None, state=None):
        if mask is None:
            mask = T.to_cuda(torch.ones(*input.size()[:-1]))

        if self.batch_first:
            input, mask = input.transpose(0,1), mask.transpose(0,1)

        batch_size = input.size(1)
        if state is None:
            state = T.to_cuda(torch.zeros(1, batch_size, self.dim_hidden))

        if self.reverse:
            input, mask = T.flip(input), T.flip(mask)

        do_dropout = (self.training and 
            self.drop_method != 'none' and self.drop_prob > 0.0)

        ZR_x, H_x = self.x2zr(input), self.x2h(input)

        output = []
        for i, (zr_x, h_x) in enumerate(zip(ZR_x, H_x)):
            m = mask[i].unsqueeze(1)
            h_ = state.squeeze(0)

            pre_zr = (zr_x + self.h2zr(h_)).sigmoid()
            z, r = torch.chunk(pre_zr, 2, dim=1)
            h = (h_x + r*self.h2h(h_)).tanh()
            h = (1 - z) * h + z * h_

            if do_dropout and self.drop_method == 'link':
                h = F.dropout(h, p=self.drop_prob, training=self.training)

            h = m * h + (1. - m) * h_

            if do_dropout and self.drop_method == 'output':
                output.append(F.dropout(h, p=self.drop_prob, training=self.training))
            else:
                output.append(h)
            state = h.unsqueeze(0)

        output = torch.stack(output)

        if self.reverse:
            output = T.flip(output)

        if self.batch_first:
            output, state = output.transpose(0, 1), state.transpose(0, 1)
        return output, state
