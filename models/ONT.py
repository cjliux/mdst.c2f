#coding: utf-8
"""
    v7: from v3
    no postr reg
    v8: gate on dec hidden
    v9: bert only for init ctx val
    v8: init ctx as hidden, slot as input
    v7: no bert; init ctx from hid
EPG =>
    v1: elmo for enc; sep vocab
    v2: no elmo
AutoBase -> C2F_A
    v1: add rank gen
    v2: add rank loss
C2F_A2 -> ONT
    v1: no gen mod
"""
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

import clks.func.tensor as T
from clks.utils.model import Model
from .vocab import WordVocab, LabelVocab
from .masked_cross_entropy import masked_cross_entropy_for_value
from clks.nnet.normalize import LayerNorm

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

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    ptr_seqs, _ = merge(item_info['ptr_context'])
    y_seqs, y_lengths = merge_multi_response(item_info["generate_y"])
    gating_label = torch.tensor(gating_dict.token2index(item_info["gating_label"]))
    turn_domain = torch.tensor(item_info["turn_domain"])

    if torch.cuda.is_available():
        src_seqs = src_seqs.cuda()
        ptr_seqs = ptr_seqs.cuda()
        gating_label = gating_label.cuda()
        turn_domain = turn_domain.cuda()
        y_seqs = y_seqs.cuda()
        y_lengths = y_lengths.cuda()

    item_info["context"] = src_seqs
    item_info["ptr_context"] = ptr_seqs
    item_info["context_len"] = src_lengths
    item_info["gating_label"] = gating_label
    item_info["turn_domain"] = turn_domain
    item_info["generate_y"] = y_seqs
    item_info["y_lengths"] = y_lengths
    return item_info


class EncoderRNN(nn.Module):
    def __init__(self, args, config, 
            word_vocab, mem_vocab, hidden_size, dropout, n_layers=1):
        super().__init__()      
        self.args, self.config = args, config
        self.word_vocab = word_vocab
        self.mem_vocab = mem_vocab
        self.vocab_size = len(word_vocab)
        self.mem_vocab_size = len(mem_vocab)

        self.hidden_size = hidden_size  
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        # emb
        self.embedding = nn.Embedding(
            self.vocab_size, 400, padding_idx=word_vocab.pad_index)
        self.embedding.weight.data.normal_(0, 0.1)

        if self.args["load_embedding"]:
            with open(os.path.join(config["word_embedding_path"])) as fd:
                E = json.load(fd)
            self.embedding.weight.data.copy_(self.embedding.weight.data.new(E))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

        if self.args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

        ## enc
        # self.elmo = ElmoAdaptor(config["elmo_option_file"], config["elmo_weight_file"],
        #     num_output_representations=1, requires_grad=False, 
        #     scalar_mix_parameters=None, dropout=self.dropout)
        # self.elmo_proj = nn.Linear(1024, 300)
        
        self.gru = nn.GRU(hidden_size, hidden_size, 
                n_layers, dropout=dropout, bidirectional=True)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return T.to_cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, data, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs) # seq-first

        # input_ids = self.elmo.process_sent(data["context_plain"])
        # elmo_feat = self.elmo(input_ids)[0].transpose(0, 1) # seq_first
        # elmo_feat = self.elmo_proj(elmo_feat)
        # embedded = torch.cat((embedded, elmo_feat), -1)

        embedded = self.dropout_layer(embedded) # seq, bat, dim

        hidden = self.get_state(input_seqs.size(1))
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths is not None:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)   
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]

        return outputs.transpose(0,1), hidden.unsqueeze(0)

    def forward_batfirst(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs) # bat-first
        embedded = self.dropout_layer(embedded) 

        hidden = self.get_state(input_seqs.size(0))

        if input_lengths is not None:
            len_sorted, idx1 = torch.sort(input_lengths, descending=True)
            emb_sorted = embedded.index_select(0, idx1).contiguous()

            embedded = nn.utils.rnn.pack_padded_sequence(
                emb_sorted, len_sorted.tolist(), batch_first=True)

        outputs, hidden = self.gru(embedded, hidden)
        
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            
            _, idx2 = torch.sort(idx1)
            outputs = outputs.index_select(0, idx2).contiguous()
            hidden = hidden.index_select(1, idx2).contiguous()

        hidden = hidden[0] + hidden[1]
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs, hidden.unsqueeze(0)



def get_vocab_map(word_vocab, mem_vocab):
    map_id = []
    for imw, mw in enumerate(mem_vocab.get_vocab()):
        iww = word_vocab.index_of(mw)
        map_id.append(iww)
    map_id = T.to_cuda(torch.Tensor(map_id).long())
    return map_id


def pad_and_merge(sequences):
    '''
    merge from batch * sent_len to batch * max_len 
    '''
    lengths = [len(seq) for seq in sequences]
    max_len = 1 if max(lengths)==0 else max(lengths)
    padded_seqs = torch.ones(len(sequences), max_len).long()
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = torch.Tensor(seq[:end])
    padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
    lengths = torch.tensor(lengths)
    return padded_seqs, lengths


class Generator(nn.Module):
    def __init__(self, args, config, word_vocab, mem_vocab, 
            ont_vocabs, encoder, shared_emb, 
            hidden_size, dropout, slots, nb_gate):
        super().__init__()
        self.args, self.config = args, config
        # self.vocab_size = len(word_vocab)
        self.mem_vocab_size = len(mem_vocab)
        self.word_vocab = word_vocab
        self.mem_vocab = mem_vocab
        
        self.mem2word_map = get_vocab_map(word_vocab, mem_vocab)

        self.encoder = encoder
        self.ont_vocabs = ont_vocabs

        ## emb
        self.src_embedding = shared_emb 
        # self.mem_embedding = nn.Embedding(len(self.mem_vocab), 
        #                 400, padding_idx=self.mem_vocab.pad_index)
        # self.mem_embedding.weight.data.normal_(0, 0.1)

        # if self.args["load_embedding"]:
        #     with open(os.path.join(config["mem_embedding_path"])) as fd:
        #         E = json.load(fd)
        #     self.mem_embedding.weight.data.copy_(self.mem_embedding.weight.data.new(E))
        #     self.mem_embedding.weight.requires_grad = True
        #     print("Encoder embedding requires_grad", self.mem_embedding.weight.requires_grad)

        # if self.args["fix_embedding"]:
        #     self.mem_embedding.weight.requires_grad = False

        self.slot_w2i = {}
        for slot in slots.get_vocab():
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)

        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3*hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        self.W_att_prior_slot = nn.Linear(hidden_size, hidden_size)
        self.W_gate = nn.Linear(hidden_size, nb_gate)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, 
            ostory, story, pstory, max_res_len, target_batches, use_teacher_forcing, slot_temp,
            postr_info = None):
        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.mem_vocab_size)
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
                domain_w2idx = T.to_cuda(torch.tensor(domain_w2idx))
                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = T.to_cuda(torch.tensor(slot_w2idx))
                slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb

        # Compute pointer-generator output, decoding each (domain, slot) one-by-one
        words_point_out = []
        all_rank_matrices = []
        kld_reg = []

        mem_embedding = self.src_embedding(self.mem2word_map)

        analyze = [{} for _ in range(ostory.size(0))]

        for islot, slot in enumerate(slot_temp):
            hidden = encoded_hidden
            words = []
            slot_emb = slot_emb_dict[slot]
            slot_ctrl = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)

            # C2F_A
            onty = [w.split() for w in self.ont_vocabs[slot].get_vocab()]
            onty_seqs = self.word_vocab.token2index(onty)
            onty_seqs, onty_lens = pad_and_merge(onty_seqs)
            _, onty_repr = self.encoder.forward_batfirst(
                T.to_cuda(onty_seqs), T.to_cuda(onty_lens)) # VD
            onty_repr = onty_repr.squeeze(0)

            cand_qry = slot_ctrl.unsqueeze(1) + onty_repr.unsqueeze(0)
            cand_att_sc = torch.bmm(cand_qry, encoded_outputs.transpose(1,2))
            for i, l in enumerate(encoded_lens):
                if l < cand_att_sc.size(-1):
                    cand_att_sc[i, :, l:] = -np.inf
            cand_att_alpha = F.softmax(cand_att_sc, -1)
            cand_ctx = torch.bmm(cand_att_alpha, encoded_outputs) # BVD

            rank_matrix = cand_ctx.mul(onty_repr.unsqueeze(0).expand_as(cand_ctx)).sum(-1)
            all_rank_matrices.append(rank_matrix)

            topk_rsc, topk_idx = torch.topk(rank_matrix, k=3, dim=1) 
            topk_ctx = cand_ctx.gather(1, 
                topk_idx.unsqueeze(2).expand(topk_idx.size(0), topk_idx.size(1), cand_ctx.size(2))) # BKD
            topk_sc = torch.bmm(topk_ctx, self.W_att_prior_slot(slot_ctrl).unsqueeze(-1)).squeeze(-1) # BK
            topk_alpha = F.softmax(topk_sc, -1)
            postr_ctx = torch.bmm(topk_alpha.unsqueeze(1), topk_ctx).squeeze(1)
            
            # ######## old code
            # prior_ctx, prior_scores = self.attend2(
            #     encoded_outputs, encoded_outputs, self.W_att_prior_slot(slot_ctrl), encoded_lens)
            # ########
            prior_ctx = postr_ctx # switch

            decoder_input, hidden = None, None

            for wi in range(max_res_len):
                if wi == 0:
                    decoder_input = slot_ctrl.contiguous()
                    hidden = prior_ctx.unsqueeze(0).contiguous()

                    dec_state, hidden = self.gru(decoder_input.unsqueeze(0), hidden)

                    all_gate_outputs[islot] = self.W_gate(hidden.squeeze(0))
                    break
                else:
                    dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

                context_vec, logits, prob = self.attend(
                    encoded_outputs, hidden.squeeze(0), encoded_lens)

                p_vocab = self.attend_vocab(mem_embedding, hidden.squeeze(0))
                p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))

                p_context_ptr = T.to_cuda(torch.zeros(p_vocab.size()))
                p_context_ptr.scatter_add_(1, pstory, prob)
                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab

                pred_word = torch.argmax(final_p_vocab, dim=1)
                all_point_outputs[islot, :, wi, :] = final_p_vocab

                words.append([self.mem_vocab.index2token(w_idx.item()) for w_idx in pred_word])

                if self.args["analyze"] and wi == 0:
                    k = 3
                    topk_prior_sc = [[(' '.join(self.word_vocab.index2token(ostory[i, w-2:w+3].clone().cpu().numpy())), v.item()) 
                            for v, w in zip(*torch.topk(sc, k))] for i, sc in enumerate(prior_scores)]
                    topk_final_w = [[(self.word_vocab.token_of(w.item()) , v.item()) for v, w in zip(tpkv, tpkw)] for tpkv, tpkw in zip(*torch.topk(final_p_vocab, k, -1))]
                    topk_ptr_w = [[(' '.join(self.word_vocab.index2token(ostory[i, w-2:w+3].clone().cpu().numpy())), v.item()) for v, w in zip(*torch.topk(sc, k))] for i, sc in enumerate(prob)]
                    topk_gen_w = [[(self.word_vocab.token_of(w.item()) , v.item()) for v, w in zip(tpkv, tpkw)] for tpkv, tpkw in zip(*torch.topk(p_vocab, k, -1))]

                    tgt_w = [' '.join(self.word_vocab.index2token([w.item() for w in inst[islot] if w not in [1, 2]])) for inst in target_batches]
                    sws = vocab_pointer_switches.view(-1).tolist()

                    gate_slot = all_gate_outputs[islot].tolist()

                    for i_inst in range(batch_size):
                        analyze[i_inst][slot] = {"ctx": topk_prior_sc[i_inst], 
                            "ptr": topk_ptr_w[i_inst], "gen": topk_gen_w[i_inst],
                            "tgt": tgt_w[i_inst], "sw": sws[i_inst], 
                            "final": topk_final_w[i_inst], "gate": gate_slot[i_inst]}
    
                if use_teacher_forcing:
                    decoder_input = mem_embedding[target_batches[:, islot, wi]] # Chosen word is next input
                else:
                    decoder_input = mem_embedding[pred_word]
                if torch.cuda.is_available(): 
                    decoder_input = decoder_input.cuda()

            # if not self.training:
            words_point_out.append(words)

        if self.args["analyze"]:
            return all_point_outputs, all_gate_outputs, words_point_out, [], analyze, all_rank_matrices
        else:
            return all_point_outputs, all_gate_outputs, words_point_out, [], all_rank_matrices

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

    def attend2(self, seqk, seqv, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seqk).mul(seqk).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seqv).mul(seqv).sum(1)
        return context, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        scores = F.softmax(scores_, dim=1)
        return scores


class ONT(Model):

    def __init__(self, args, config):
        super().__init__()
        self.args, self.config = args, config
        self.hidden_size = int(args["hidden"])
        self.word_vocab = WordVocab(config["word_vocab"])
        self.mem_vocab = WordVocab(config["mem_vocab"])
        self.gating_dict = LabelVocab(config["gating_dict"])
        self.domain_dict = LabelVocab(config["domain_dict"])
        self.all_slots_dict = LabelVocab(config["all_slots_dict"])
        self.ont_vocabs = { slot: LabelVocab(config["ont_vocabs_fmt"].format(slot))
                                for slot in self.all_slots_dict.get_vocab() }

        self.nb_gate = len(self.gating_dict)
        self.dropout = float(args["drop"])
        self.cross_entropy = nn.CrossEntropyLoss()

        self.encoder = EncoderRNN(args, config, 
            self.word_vocab, self.mem_vocab,
            self.hidden_size, self.dropout)
        self.decoder = Generator(args, config, 
            self.word_vocab, self.mem_vocab, self.ont_vocabs,
            self.encoder, self.encoder.embedding, 
            self.hidden_size, self.dropout, 
            self.all_slots_dict, self.nb_gate)

        # self.beta_epo = 0.

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
        ptr_story = data["ptr_context"]

        # Encode dialog history
        encoded_outputs, encoded_hidden = self.encoder(
                        data, story.transpose(0, 1), data['context_len'])

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10

        # TODO: enc y
        if self.training:
            postr_info = None

            (all_point_outputs, all_gate_outputs, words_point_out, 
                words_class_out, all_rank_matrices) = self.decoder.forward(batch_size, 
                    encoded_hidden, encoded_outputs, data['context_len'], 
                    data["context"], story, ptr_story, max_res_len, data['generate_y'], 
                    use_teacher_forcing, slot_temp, postr_info) 
            return all_point_outputs, all_gate_outputs, words_point_out, words_class_out, all_rank_matrices
        else:
            if self.args["analyze"]:
                (all_point_outputs, all_gate_outputs, words_point_out, words_class_out, 
                        analyze, all_rank_matrices) = self.decoder.forward(batch_size, 
                    encoded_hidden, encoded_outputs, data['context_len'], 
                    data["context"], story, ptr_story, max_res_len, data['generate_y'], 
                    use_teacher_forcing, slot_temp) 
                return all_point_outputs, all_gate_outputs, words_point_out, words_class_out, analyze, all_rank_matrices
            else:
                (all_point_outputs, all_gate_outputs, words_point_out, words_class_out, 
                        all_rank_matrices) = self.decoder.forward(batch_size, 
                    encoded_hidden, encoded_outputs, data['context_len'], 
                    data["context"], story, ptr_story, max_res_len, data['generate_y'], 
                    use_teacher_forcing, slot_temp) 
                return all_point_outputs, all_gate_outputs, words_point_out, words_class_out, all_rank_matrices

    def preprocess_data(self, data):
        data["turn_domain"] = [preprocess_domain(item, self.domain_dict) for item in data["turn_domain"]]
        data["gen_y_plain"] = [[' '.join(y) for y in ys] for ys in data["generate_y"]]
        data["generate_y"] = [preprocess_slot(item, self.mem_vocab) for item in data["generate_y"]]
        data["context"] = [preprocess(item, self.word_vocab) for item in data["context_plain"]]
        data["ptr_context"] = [preprocess(item, self.mem_vocab) for item in data["context_plain"]]
        data = collate_fn(data, self.word_vocab, self.gating_dict)
        return data

    def masked_cross_entropy_for_value(self, prob, generate_y, y_lengths):
        return masked_cross_entropy_for_value(prob, generate_y, y_lengths)
        # log_prob = prob.log()
        # nll = - log_prob.gather(3, generate_y.unsqueeze(-1)).squeeze(-1)
        # msk_y = T.to_cuda(torch.arange(nll.size(-1))).expand_as(nll) < y_lengths.unsqueeze(-1)
        # n_tokens = msk_y.sum().float()
        # loss = nll[msk_y].sum() / msk_y.sum()
        # return loss

    def ranking_loss(self, rank_matrices, gen_y_plain, slot_temp):
        rank_loss = None
        for islot, slot in enumerate(slot_temp):
            onty_label = T.to_cuda(torch.Tensor(self.ont_vocabs[slot].token2index(
                                        [gen_y[islot] for gen_y in gen_y_plain])).long())
            rank_matrix = rank_matrices[islot]
            target_rank_score = rank_matrix.gather(1, onty_label.unsqueeze(-1))
            res_matrix = rank_matrix.masked_fill(
                T.to_cuda(torch.arange(rank_matrix.size(-1))).expand_as(rank_matrix) == onty_label.unsqueeze(-1), -1e9)
            contr_score = res_matrix.max(1)[0]
            contr_loss = torch.relu(1 - target_rank_score + contr_score).mean()
            rank_loss = contr_loss if rank_loss is None else rank_loss + contr_loss
        return rank_loss

    def compute_loss(self, data):
        data = self.preprocess_data(data)

        use_teacher_forcing = random.random() < self.args["teacher_forcing_ratio"]
        slot_temp = data["slot_temp"][0]
        (all_point_outputs, gates, words_point_out, 
            words_class_out, all_rank_matrices) = self.encode_and_decode(
                    data, use_teacher_forcing, slot_temp)

        # loss_ptr = self.masked_cross_entropy_for_value(
        #     all_point_outputs.transpose(0, 1).contiguous(),
        #     data["generate_y"].contiguous(), #[:,:len(self.point_slots)].contiguous(),
        #     data["y_lengths"]) #[:,:len(self.point_slots)])
        loss_gate = self.cross_entropy(
            gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)), 
            data["gating_label"].contiguous().view(-1))
        loss_rank = self.ranking_loss(
            all_rank_matrices, data["gen_y_plain"], slot_temp)

        if self.args["use_gate"]:
            loss = loss_gate + loss_rank
        else:
            loss = loss_rank

        return loss, { "LG": loss_gate, "LR": loss_rank }

    def inference(self, data):
        pass

    def trainable_parameters(self):
        # return filter(lambda p: p.requires_grad, self.parameters())
        return self.parameters()

    def before_epoch(self, curr_epoch):
        # if curr_epoch < 12:
        #     nepo_per_cycle = 3
        #     tao = (curr_epoch % nepo_per_cycle) /  nepo_per_cycle
        #     R = 0.6
        #     self.beta_epo = min(tao / R, 1)
        # else:
        #     self.beta_epo = 1.
        pass
