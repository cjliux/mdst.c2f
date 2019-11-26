#coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn  as sns
# import nltk
import os
import json
# import pandas as pd
import copy
import math
import pprint
import torch.nn.utils.rnn as rnn_utils

import clks.func.tensor as T
from clks.utils.model import Model
from .vocab import WordVocab, LabelVocab
from .masked_cross_entropy import masked_cross_entropy_for_value
from clks.nnet.normalize import LayerNorm
from clks.nnet.rnn import LSTMCell, GRUCell, TPGRUCell


###
# Bert
###
class BertSelfAttention(nn.Module):

    def __init__(self, dim_hidden, num_heads, dropout):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.head_size = int(dim_hidden / num_heads)
        self.all_head_size = num_heads * self.head_size

        self.query = nn.Linear(dim_hidden, self.all_head_size)
        self.key = nn.Linear(dim_hidden, self.all_head_size)
        self.value = nn.Linear(dim_hidden, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attn_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_size)
        attn_scores = attn_scores + attn_mask

        attn_probs = F.softmax(attn_scores, -1)
        attn_probs = self.dropout(attn_probs)

        ctx_layer = torch.matmul(attn_probs, value_layer)
        ctx_layer = ctx_layer.permute(0,2,1,3).contiguous()
        new_ctx_layer_shape = ctx_layer.size()[:-2] + (self.all_head_size,)
        ctx_layer = ctx_layer.view(*new_ctx_layer_shape)
        return ctx_layer


class BertSelfOutput(nn.Module):

    def __init__(self, dim_hidden, dropout):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dense = nn.Linear(dim_hidden, dim_hidden)
        self.layer_norm = LayerNorm(dim_hidden, learnable=True, epsilon=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, dim_hidden, num_heads, dropout, batch_first=False):
        super().__init__()
        self.self = BertSelfAttention(dim_hidden, num_heads, dropout)
        self.output = BertSelfOutput(dim_hidden, dropout)
        self.batch_first = batch_first

    def forward(self, input_tensor, attn_mask):
        """
        input_tensor: T(bat, seq, dim)
        # attn_mask: T()
        input_lengths: L(bat)
        """
        if not self.batch_first:
            input_tensor = input_tensor.transpose(0,1)
        
        self_output = self.self(input_tensor, attn_mask)
        attn_output = self.output(self_output, input_tensor)

        if not self.batch_first:
            attn_output = attn_output.transpose(0,1)
        return attn_output


class BertIntermediate(nn.Module):

    def __init__(self, dim_hidden, dim_inter):
        super().__init__()
        self.dim_hidden, self.dim_inter = dim_hidden, dim_inter
        self.dense = nn.Linear(dim_hidden, dim_inter)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    
    def __init__(self, dim_inter, dim_hidden, dropout):
        super().__init__()
        self.dim_inter, self.dim_hidden = dim_inter, dim_hidden
        self.dense = nn.Linear(dim_inter, dim_hidden)
        self.layer_norm = LayerNorm(dim_hidden, learnable=True, epsilon=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, dim_hidden, num_heads, dim_inter, dropout, batch_first):
        super().__init__()
        self.attention = BertAttention(dim_hidden, num_heads, dropout, batch_first=True)
        self.intermediate = BertIntermediate(dim_hidden, dim_inter)
        self.output = BertOutput(dim_inter, dim_hidden, dropout)
        self.batch_first = batch_first
    
    def forward(self, hidden_states, attn_mask):
        if not self.batch_first:
            hidden_states = hidden_states.transpose(0,1)
        attn_output = self.attention(hidden_states, attn_mask)
        inter_output = self.intermediate(attn_output)
        layer_output = self.output(inter_output, attn_output)
        if not self.batch_first:
            layer_output = layer_output.transpose(0,1)
        return layer_output


class BertEncoder(nn.Module):

    def __init__(self, n_layers, 
            dim_hidden, num_heads, dim_inter, dropout, batch_first):
        super().__init__()
        self.n_layers = n_layers
        self.layer = BertLayer(dim_hidden, num_heads, dim_inter, dropout, batch_first=True)
        self.batch_first = batch_first
    
    def forward(self, hidden_states, input_lengths):
        if not self.batch_first:
            hidden_states = hidden_states.transpose(0,1)

        attn_mask = torch.zeros(hidden_states.size()[:2])
        if torch.cuda.is_available(): attn_mask = attn_mask.cuda()
        for i, l in enumerate(input_lengths):
            attn_mask[i,l:] = -1e9
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        layer_output = hidden_states
        for i_layer in range(self.n_layers): 
            layer_output = self.layer(layer_output, attn_mask)

        if not self.batch_first:
            layer_output = layer_output.transpose(0,1)
        return layer_output

###
# TM
###
class TopicModel(nn.Module):

    def __init__(self, word_vocab, embedding, dim_latent, lamda, n_negs=None):
        super().__init__()
        self.word_vocab = word_vocab
        self.dim_embed = embedding.embedding_dim
        self.dim_latent = dim_latent
        self.embedding = embedding
        self.n_negs = n_negs # only for intro loss
        self.lamda = lamda
        self.M = nn.Linear(self.dim_embed, self.dim_embed, bias=False)
        self.W = nn.Linear(self.dim_embed, dim_latent, bias=False)
        self.T = nn.Linear(dim_latent, self.dim_embed, bias=False)

    def forward(self, input_tensor, input_lengths, segments):
        """
        input_tensor: T(BS) -> id
        """
        max_num_seq = max(len(s) for s in segments)
        yzpr_list = []
        for ib, (ten, seg) in enumerate(zip(input_tensor, segments)):
            seq_bat = []
            max_len = max(e-s for s,e in seg)
            sub_ten = torch.zeros((len(seg), max_len)).cuda()
            sub_msk = torch.zeros((len(seg), max_len)).cuda()
            for iseg, (start, end) in enumerate(seg):
                sub_ten[iseg,:(end-start)] = ten[start:end]
                sub_msk[iseg,:(end-start)] = 1.
            sub_ten = sub_ten.long()
            sub_emb = self.embedding(sub_ten)
            # sub_emb = sub_emb.clone().detach()
            y = torch.sum(sub_emb * sub_msk.unsqueeze(-1), 1) / torch.sum(sub_msk, 1).unsqueeze(-1)
            esc = torch.matmul(self.M(sub_emb), y.unsqueeze(-1)).squeeze(-1)
            esc.masked_fill_(sub_msk == 0, -1e9)
            alpha = F.softmax(esc, 1)
            z = torch.matmul(alpha.unsqueeze(1), sub_emb).squeeze(1)
            p = F.softmax(self.W(z), -1)
            r = self.T(p)
            yzpr_list.append((y, z, p, r))
        return yzpr_list

    def compute_intro_loss(self, yzpr_list):
        # zpr_list = self.forward(input_tensor, input_lengths, segments)
        y_mat, z_mat, p_mat, r_mat = list(zip(*yzpr_list))
        y_mat = torch.cat(y_mat, 0)
        z_mat = torch.cat(z_mat, 0)
        # p_mat = torch.cat(p_mat, 0)
        r_mat = torch.cat(r_mat, 0)
        rz_mat = torch.sum(r_mat * z_mat, -1)
        ry_mat = torch.matmul(r_mat, y_mat.t())
        lcon = F.relu(1 - rz_mat.unsqueeze(-1) + ry_mat)
        lcon = lcon.masked_fill(T.to_cuda(torch.eye(rz_mat.size(0)).byte()), -1e-9)
        if self.n_negs is not None:
            lcon = torch.topk(lcon, min(self.n_negs, lcon.size(1)-1), 1)[0]
        # J = torch.mean(torch.sum(lcon, 1))
        J = torch.sum(lcon)
        TTT = torch.matmul(self.T.weight.t(), self.T.weight)
        U = torch.norm(TTT - T.to_cuda(torch.eye(TTT.size(0))), 'fro')
        L = J + self.lamda * U
        return L, {"TJ": J, "TU": U}

    def compute_contra_loss(self, yzpr_list, neg_tensor, neg_lengths):
        """neg_insts: T(BNS)"""
        _, z_mat, p_mat, r_mat = list(zip(*yzpr_list))
        # z_mat, r_mat = torch.cat(z_mat, 0), torch.cat(r_mat, 0)
        z_mat, r_mat = z_mat[-1], r_mat[-1]
        rz_mat = torch.sum(r_mat * z_mat, -1)
        n = []
        for neg_ten, neg_len in zip(neg_tensor, neg_lengths):
            neg_msk = T.to_cuda(torch.zeros(*neg_ten.size()))
            neg_emb = self.embedding(neg_ten) # NSD
            n.append(torch.sum(neg_emb * neg_msk.unsqueeze(-1), 1) 
                                        / neg_msk.sum(1).unsqueeze(-1))
        n_mat = torch.stack(n) # BND
        rn_mat = torch.matmul(n_mat, r_mat.unsqueeze(-1)).squeeze(-1)
        lcon = F.relu(1 - rz_mat.unsqueeze(-1) + rn_mat)
        # lcon = torch.topk(lcon, min(self.n_negs, lcon.size(1)), 1)[0]
        J = torch.sum(lcon)
        TTT = torch.matmul(self.T.weight.t(), self.T.weight)
        U = torch.norm(TTT - T.to_cuda(torch.eye(TTT.size(0))), 'fro')
        L = J + self.lamda * U
        return L, {"TJ": J, "TU": U}

    def save_topk_topic_words(self, file_name, topk=None):
        rel = torch.matmul(self.embedding.weight, self.T.weight)
        if topk is None: topk = len(self.word_vocab)
        _, tk = torch.topk(rel, topk, 0, largest=True)
        tws = self.word_vocab.index2token(tk.t().tolist())
        with open(file_name, 'w', encoding='utf8') as fd:
            for itp, tpw in enumerate(tws):
                fd.write("Topic {}: ".format(itp) + "\n")
                nln = len(tpw) // 15
                for i in range(nln):
                    fd.write(" ".join(tpw[i*15:(i+1)*15]) + "\n")
                if nln * 15 < len(tpw):
                    fd.write(" ".join(tpw[nln*15:]) + "\n")
                fd.write("\n")

    def save_exclusive_topic_words(self, file_name):
        rel = torch.matmul(self.embedding.weight, self.T.weight)
        sccls, tpcls = torch.max(rel, 1)
        with open(file_name, 'w', encoding='utf8') as fd:
            for itp in range(self.dim_latent):
                fd.write("Topic {}: ".format(itp) + "\n")
                tpws = torch.nonzero(tpcls == itp).view(-1)
                if len(tpws) > 0: 
                    subsc = sccls[tpws]
                    _, idx = torch.sort(subsc, descending=True)
                    tpws = self.word_vocab.index2token(tpws[idx].tolist())
                    for iw, w in enumerate(tpws):
                        fd.write(w + " ")
                        if (iw + 1) % 15 == 0: fd.write('\n')
                fd.write('\n')


def extract_segments(data):
    segments = []
    for ctx_pln in data['context_plain']:
        sub_seg = []
        start, end, hit = None, 0, 0
        for ic, c in enumerate(ctx_pln):
            if c == ';':
                hit += 1
                if hit % 2 == 0:
                    start, end = end, ic + 1
                    sub_seg.append((start, end))
        segments.append(sub_seg)
    return segments

###
# original
###

def preprocess(sequence, vocab):
    """Converts words to ids."""
    story = vocab.token2index(sequence)
    story = torch.Tensor(story)
    return story


def preprocess_slot(sequence, vocab):
    """Converts words to ids."""
    story = []
    for value in sequence:
        v = vocab.token2index(value) + [vocab.eos_index]
        story.append(v)
    # story = torch.Tensor(story)
    return story


def preprocess_domain(turn_domain, domain_dict):
    return domain_dict.index_of(turn_domain)


def collate_fn(item_info, vocab, gating_dict):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [vocab.pad_index] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def merge_memory(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths) # avoid the empty belief state issue
        padded_seqs = torch.ones(len(sequences), max_len, 4).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i,:end,:] = seq[:end]
        return padded_seqs, lengths

    # # sort a list by sequence length (descending order) to use pack_padded_sequence
    # data.sort(key=lambda x: len(x['context']), reverse=True) 
    # item_info = {}
    # for key in data[0].keys():
    #     item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    y_seqs, y_lengths = merge_multi_response(item_info["generate_y"])
    gating_label = torch.tensor(gating_dict.token2index(item_info["gating_label"]))
    turn_domain = torch.tensor(item_info["turn_domain"])

    neg_tensor, neg_lengths = [], []
    for neg_ten in item_info["neg_tensor"]:
        seqs, lens = merge(neg_ten)
        neg_tensor.append(T.to_cuda(seqs))
        neg_lengths.append(lens)

    if torch.cuda.is_available():
        src_seqs = src_seqs.cuda()
        gating_label = gating_label.cuda()
        turn_domain = turn_domain.cuda()
        y_seqs = y_seqs.cuda()
        y_lengths = y_lengths.cuda()

    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["gating_label"] = gating_label
    item_info["turn_domain"] = turn_domain
    item_info["generate_y"] = y_seqs
    item_info["y_lengths"] = y_lengths
    item_info["neg_tensor"] = neg_tensor
    item_info["neg_lengths"] = neg_lengths
    return item_info


class EncoderRNN(nn.Module):
    def __init__(self, args, config, 
            word_vocab, hidden_size, dropout, n_layers=1):
        super().__init__()      
        self.args, self.config = args, config
        self.vocab_size = len(word_vocab)
        self.hidden_size = hidden_size  
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            self.vocab_size, hidden_size, padding_idx=word_vocab.pad_index)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

        self.bert_enc = BertEncoder(1, hidden_size, 4, hidden_size, dropout, batch_first=False)

        if self.args["load_embedding"]:
            with open(os.path.join(config["embedding_path"])) as fd:
                E = json.load(fd)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

        if self.args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

        # print("encoder params:")
        # for name, param in self.named_parameters():
        #     print(name)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return T.to_cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None, source=True):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs) # seq-first
        embedded = self.dropout_layer(embedded) # seq, bat, dim
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)   
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]

        ## added
        if source: outputs = self.bert_enc(outputs, input_lengths)
        return outputs.transpose(0,1), hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, args, config, word_vocab, shared_emb, 
            hidden_size, dropout, slots, nb_gate):
        super().__init__()
        self.args, self.config = args, config
        self.vocab_size = len(word_vocab)
        self.word_vocab = word_vocab
        self.embedding = shared_emb 
        self.dropout_layer = nn.Dropout(dropout)
        # self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.gru = TPGRUCell(hidden_size, hidden_size, args["num_topic"],
            drop_method='output', drop_prob=dropout)

        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3*hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        # self.W_att_key = nn.Linear(hidden_size + args["num_topic"], hidden_size)
        self.W_att_prior_slot = nn.Linear(hidden_size, hidden_size)
        self.W_att_postr_slot = nn.Linear(hidden_size, hidden_size)
        self.W_gate = nn.Linear(hidden_size, nb_gate)
        # edg v3: topic gate
        self.W_gate_topic = nn.Linear(args["num_topic"], nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots.get_vocab():
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

        # print("decoder params:")
        # for name, param in self.named_parameters():
        #     print(name)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, 
            story, max_res_len, target_batches, use_teacher_forcing, slot_temp,
            yzpr_list, segments, postr_info = None):
        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size)
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate)
        if torch.cuda.is_available(): 
            all_point_outputs = all_point_outputs.cuda()
            all_gate_outputs = all_gate_outputs.cuda()
        
        # Get the slot embedding 
        slot_emb_dict = {}
        for i, slot in enumerate(slot_temp):
            # Domain embbeding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                if torch.cuda.is_available(): domain_w2idx = domain_w2idx.cuda()
                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx)
                if torch.cuda.is_available(): slot_w2idx = slot_w2idx.cuda()
                slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
            slot_emb_exp = combined_emb.expand_as(encoded_hidden)
            if i == 0:
                slot_emb_arr = slot_emb_exp.clone()
            else:
                slot_emb_arr = torch.cat((slot_emb_arr, slot_emb_exp), dim=0)

        # get topic dist
        _, _, p_mat, _ = list(zip(*yzpr_list))

        # if self.args["parallel_decode"]:
        #     # Compute pointer-generator output, puting all (domain, slot) in one batch
        #     decoder_input = self.dropout_layer(slot_emb_arr).view(-1, self.hidden_size) # (batch*|slot|) * emb
        #     hidden = encoded_hidden.repeat(1, len(slot_temp), 1) # 1 * (batch*|slot|) * emb
        #     words_point_out = [[] for i in range(len(slot_temp))]
        #     words_class_out = []

        #     for wi in range(max_res_len):
        #         dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

        #         enc_out = encoded_outputs.repeat(len(slot_temp), 1, 1)
        #         enc_len = encoded_lens * len(slot_temp)
        #         context_vec, logits, prob = self.attend(enc_out, hidden.squeeze(0), enc_len)

        #         if wi == 0: 
        #             all_gate_outputs = torch.reshape(self.W_gate(context_vec), all_gate_outputs.size())

        #         p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
        #         p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
        #         vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                
        #         p_context_ptr = torch.zeros(p_vocab.size())
        #         if torch.cuda.is_available(): 
        #             p_context_ptr = p_context_ptr.cuda()
        #         p_context_ptr.scatter_add_(1, story.repeat(len(slot_temp), 1), prob)

        #         final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
        #                         vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
        #         pred_word = torch.argmax(final_p_vocab, dim=1)
                
        #         words = [self.word_vocab.index2token(w_idx.item()) for w_idx in pred_word]

        #         for si in range(len(slot_temp)):
        #             words_point_out[si].append(words[si*batch_size:(si+1)*batch_size])
                
        #         all_point_outputs[:, :, wi, :] = torch.reshape(final_p_vocab, 
        #                                 (len(slot_temp), batch_size, self.vocab_size))
                
        #         if use_teacher_forcing:
        #             decoder_input = self.embedding(torch.flatten(target_batches[:, :, wi].transpose(1,0)))
        #         else:
        #             decoder_input = self.embedding(pred_word)   
                
        #         if torch.cuda.is_available(): decoder_input = decoder_input.cuda()
        # else:
        if True:
            # Compute pointer-generator output, decoding each (domain, slot) one-by-one
            words_point_out = []
            kld_reg = []

            msk_input = torch.zeros(batch_size, encoded_outputs.size(1)).cuda()
            for i, l in enumerate(encoded_lens):
                msk_input[i, :l] = 1.

            # topic guided attention 
            max_num_seg = max(len(s) for s in segments)
            max_seq_len = encoded_outputs.size(1)
            topic_repr = []
            for i_bat, (sub_p, segs) in enumerate(zip(p_mat, segments)):
                seg_id = []
                for i_seg, (start, end) in enumerate(segs):
                    seg_id.extend((end-start) * [i_seg])
                seg_id = T.to_cuda(torch.Tensor(seg_id).long())
                sub_p_mat = sub_p[seg_id]
                sub_p_mat = torch.cat((sub_p_mat, T.to_cuda(torch.zeros(
                        max_seq_len-sub_p_mat.size(0), sub_p_mat.size(1)))), 0)
                topic_repr.append(sub_p_mat)
            topic_repr = torch.stack(topic_repr)

            encoded_outputs_val = torch.cat((encoded_outputs, topic_repr), -1)
            # encoded_outputs_key = self.W_att_key(encoded_outputs_val)
            encoded_outputs_key = encoded_outputs

            # incorporate topic guidance into attn
            for islot, slot in enumerate(slot_temp):
                hidden = encoded_hidden # SBD
                words = []
                slot_emb = slot_emb_dict[slot]
                slot_ctrl = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)

                # att from slot emb to enc out
                prior_expsc = torch.matmul(encoded_outputs_key, 
                    self.W_att_prior_slot(slot_ctrl).unsqueeze(-1)).squeeze(-1)
                prior_expsc.masked_fill_(msk_input == 0, -1e9)
                prior_alpha = F.softmax(prior_expsc, -1)
                prior_ctx = torch.matmul(prior_alpha.unsqueeze(1), encoded_outputs_val).squeeze(1)
                prior_ctx, prior_tp = torch.split(prior_ctx, self.hidden_size, -1)

                # postr att
                if self.training:
                    assert postr_info is not None
                    enc_y = postr_info["enc_y"] # 32 7 400
                    enc_y_slot = torch.select(enc_y, 1, islot)

                    # add postr info
                    postr_expsc = torch.matmul(encoded_outputs_key, 
                            (self.W_att_postr_slot(slot_ctrl) + enc_y_slot).unsqueeze(-1)
                        ).squeeze(-1)
                    postr_expsc.masked_fill_(msk_input == 0, -1e9)
                    postr_alpha = F.softmax(postr_expsc, -1)
                    postr_ctx = torch.matmul(postr_alpha.unsqueeze(1), encoded_outputs_val).squeeze(1)
                    postr_ctx, postr_tp = torch.split(postr_ctx, self.hidden_size, -1)

                    ## reg loss
                    kld = torch.mean(torch.sum(postr_alpha * (
                            F.log_softmax(postr_expsc, -1) - F.log_softmax(prior_expsc, -1)
                        ), -1))
                    kld_reg.append(kld)

                # all_gate_outputs[islot] = self.W_gate(prior_ctx)
                if self.training:
                    all_gate_outputs[islot] = self.W_gate(postr_ctx) * self.W_gate_topic(postr_tp)
                    hidden = postr_ctx.expand_as(hidden)
                    topic = postr_tp
                else:
                    all_gate_outputs[islot] = self.W_gate(prior_ctx) * self.W_gate_topic(prior_tp)
                    hidden = prior_ctx.expand_as(hidden)
                    topic = prior_tp
                # decoder_input = slot_ctrl
                decoder_input = None

                # topic guided reconstruction
                for wi in range(max_res_len):
                    if wi == 0:
                        decoder_input = hidden.squeeze(0)
                        dec_state, hidden = self.gru(hidden, topic)
                    else:
                        dec_state, hidden = self.gru(
                            decoder_input.expand_as(hidden), topic, state=hidden)

                    context_vec, logits, prob = self.attend(
                        encoded_outputs, hidden.squeeze(0), encoded_lens)

                    p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                    p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                    vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))

                    p_context_ptr = torch.zeros(p_vocab.size())
                    if torch.cuda.is_available(): 
                        p_context_ptr = p_context_ptr.cuda()
                    p_context_ptr.scatter_add_(1, story, prob)
                    final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                    vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab

                    pred_word = torch.argmax(final_p_vocab, dim=1)
                    words.append([self.word_vocab.index2token(w_idx.item()) for w_idx in pred_word])
                    all_point_outputs[islot, :, wi, :] = final_p_vocab

                    if use_teacher_forcing:
                        decoder_input = self.embedding(target_batches[:, islot, wi]) # Chosen word is next input
                    else:
                        decoder_input = self.embedding(pred_word)
                    if torch.cuda.is_available(): 
                        decoder_input = decoder_input.cuda()

                words_point_out.append(words)

        if self.training:
            kld_reg_loss = torch.stack(kld_reg).mean()
            return all_point_outputs, all_gate_outputs, words_point_out, [], kld_reg_loss
        else:
            return all_point_outputs, all_gate_outputs, words_point_out, []

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        scores = F.softmax(scores_, dim=1)
        return scores


class SAEe3tmEDg2N(Model):

    def __init__(self, args, config):
        super().__init__()
        self.args, self.config = args, config
        self.hidden_size = int(args["hidden"])
        self.word_vocab = WordVocab(config["word_vocab"])
        self.mem_vocab = WordVocab(config["mem_vocab"])
        self.gating_dict = LabelVocab(config["gating_dict"])
        self.domain_dict = LabelVocab(config["domain_dict"])
        self.all_slots_dict = LabelVocab(config["all_slots_dict"])
        # temperal solution
        # self.slots_train_dict = LabelVocab().from_list(slots_list[1])
        # self.slots_dev_dict = LabelVocab().from_list(slots_list[2])
        # self.slots_test_dict = LabelVocab().from_list(slots_list[3])

        self.nb_gate = len(self.gating_dict)
        self.dropout = float(args["drop"])
        self.cross_entropy = nn.CrossEntropyLoss()

        self.encoder = EncoderRNN(args, config, self.word_vocab, 
            self.hidden_size, self.dropout)
        self.decoder = Generator(args, config, 
            self.word_vocab, self.encoder.embedding, 
            self.hidden_size, self.dropout, 
            self.all_slots_dict, self.nb_gate)
        self.tm = TopicModel(self.word_vocab, 
            self.encoder.embedding, args["num_topic"], 1) # TODO: n_negs as cmd arg

        self.beta_epo = 0.

    def init_weights(self):
        pass

    def encode_and_decode(self, data, use_teacher_forcing, slot_temp=None):
        slot_temp = data["slot_temp"][0]

        # Build unknown mask for memory to encourage generalization
        if self.args['unk_mask'] and self.decoder.training:
            story_size = data['context'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))], 1-self.dropout)[0]
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.Tensor(rand_mask)
            rand_mask = T.to_cuda(rand_mask)
            story = data['context'] * rand_mask.long()
        else:
            story = data['context']

        # Encode dialog history
        encoded_outputs, encoded_hidden = self.encoder(
                            story.transpose(0, 1), data['context_len'])

        yzpr_list = self.tm(story, data["context_len"], data["segments"])

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10

        # TODO: enc y
        if self.training:
            postr_info = {}
            size_y = data['generate_y'].size()
            gen_y = data['generate_y'].contiguous().view(np.prod(size_y[:-1]), size_y[-1])
            y_lens = data['y_lengths'].contiguous().view(-1)
            y_lens_sort, idx1 = torch.sort(y_lens, descending=True)
            gen_y_sort = gen_y.index_select(0, idx1).contiguous()
            _, y_hidden = self.encoder(gen_y_sort.transpose(0,1), y_lens_sort.tolist(), source=False)
            y_hidden = y_hidden.squeeze(0) # BD
            _, idx2 = torch.sort(idx1)
            enc_y = y_hidden.index_select(0, idx2).contiguous().view(*size_y[:-1], -1)
            postr_info['enc_y'] = enc_y

            (all_point_outputs, all_gate_outputs, words_point_out, 
                words_class_out, kld_reg_loss) = self.decoder.forward(batch_size, 
                    encoded_hidden, encoded_outputs, data['context_len'], 
                    story, max_res_len, data['generate_y'], 
                    use_teacher_forcing, slot_temp, 
                    yzpr_list, data["segments"], postr_info) 
            # L_tm, disp_tmvals = self.tm.compute_loss(zpr_list)
            return all_point_outputs, all_gate_outputs, words_point_out, words_class_out, kld_reg_loss, yzpr_list
        else:
            (all_point_outputs, all_gate_outputs, words_point_out, 
                words_class_out) = self.decoder.forward(batch_size, 
                    encoded_hidden, encoded_outputs, data['context_len'], 
                    story, max_res_len, data['generate_y'], 
                    use_teacher_forcing, slot_temp, yzpr_list, data["segments"]) 
            return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def preprocess_data(self, data):
        # construct segments
        data['segments'] = extract_segments(data)
        # conventional
        data["turn_domain"] = [preprocess_domain(item, self.domain_dict) for item in data["turn_domain"]]
        data["generate_y"] = [preprocess_slot(item, self.word_vocab) for item in data["generate_y"]]
        data["context"] = [preprocess(item, self.word_vocab) for item in data["context_plain"]]
        data["neg_tensor"] = [[ preprocess(neg.split(), self.word_vocab) for neg in negs] for negs in data["negs_plain"]]
        data = collate_fn(data, self.word_vocab, self.gating_dict)
        return data

    def compute_loss(self, data):
        data = self.preprocess_data(data)

        use_teacher_forcing = random.random() < self.args["teacher_forcing_ratio"]
        slot_temp = data["slot_temp"][0]
        (all_point_outputs, gates, words_point_out, words_class_out, 
            kld_reg_loss, zpr_list) = self.encode_and_decode(
                                    data, use_teacher_forcing, slot_temp)

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            data["generate_y"].contiguous(), #[:,:len(self.point_slots)].contiguous(),
            data["y_lengths"]) #[:,:len(self.point_slots)])
        loss_gate = self.cross_entropy(
            gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)), 
            data["gating_label"].contiguous().view(-1))

        # pack neg instances
        L_tm, tmdisp = self.tm.compute_contra_loss(
            zpr_list, data["neg_tensor"], data["neg_lengths"])

        if self.args["use_gate"]:
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr

        loss = loss + L_tm + self.beta_epo * kld_reg_loss
 
        return loss, { "LP": loss_ptr, "LG": loss_gate, 
                "BT": self.beta_epo, "KL": kld_reg_loss, "TM": L_tm,
                "TJ": tmdisp["TJ"], "TU": tmdisp["TU"] }

    def inference(self, data):
        pass

    def trainable_parameters(self):
        # return filter(lambda p: p.requires_grad, self.parameters())
        return self.parameters()

    def before_epoch(self, curr_epoch):
        if curr_epoch < 12:
            nepo_per_cycle = 3
            tao = (curr_epoch % nepo_per_cycle) /  nepo_per_cycle
            R = 0.6
            self.beta_epo = min(tao / R, 1)
        else:
            self.beta_epo = 1.

