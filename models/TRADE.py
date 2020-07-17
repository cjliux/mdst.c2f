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
import pprint

import clks.func.tensor as T
from clks.utils.model import Model
from .vocab import WordVocab, LabelVocab
from .masked_cross_entropy import masked_cross_entropy_for_value


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
    return item_info


class EncoderRNN(nn.Module):
    def __init__(self, args, config, 
            word_vocab, hidden_size, dropout, n_layers=1):
        super(EncoderRNN, self).__init__()      
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
        
    def forward(self, input_seqs, input_lengths, hidden=None):
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
        return outputs.transpose(0,1), hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, args, config, word_vocab, shared_emb, 
            hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        self.args, self.config = args, config
        self.vocab_size = len(word_vocab)
        self.word_vocab = word_vocab
        self.embedding = shared_emb 
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3*hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        self.W_gate = nn.Linear(hidden_size, nb_gate)

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
            story, max_res_len, target_batches, use_teacher_forcing, slot_temp):
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

        if self.args["parallel_decode"]:
            # Compute pointer-generator output, puting all (domain, slot) in one batch
            decoder_input = self.dropout_layer(slot_emb_arr).view(-1, self.hidden_size) # (batch*|slot|) * emb
            hidden = encoded_hidden.repeat(1, len(slot_temp), 1) # 1 * (batch*|slot|) * emb
            words_point_out = [[] for i in range(len(slot_temp))]
            words_class_out = []
            
            for wi in range(max_res_len):
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

                enc_out = encoded_outputs.repeat(len(slot_temp), 1, 1)
                enc_len = encoded_lens * len(slot_temp)
                context_vec, logits, prob = self.attend(enc_out, hidden.squeeze(0), enc_len)

                if wi == 0: 
                    all_gate_outputs = torch.reshape(self.W_gate(context_vec), all_gate_outputs.size())

                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                
                p_context_ptr = torch.zeros(p_vocab.size())
                if torch.cuda.is_available(): 
                    p_context_ptr = p_context_ptr.cuda()
                p_context_ptr.scatter_add_(1, story.repeat(len(slot_temp), 1), prob)

                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)
                
                words = [self.word_vocab.index2token(w_idx.item()) for w_idx in pred_word]

                for si in range(len(slot_temp)):
                    words_point_out[si].append(words[si*batch_size:(si+1)*batch_size])
                
                all_point_outputs[:, :, wi, :] = torch.reshape(final_p_vocab, 
                                        (len(slot_temp), batch_size, self.vocab_size))
                
                if use_teacher_forcing:
                    decoder_input = self.embedding(torch.flatten(target_batches[:, :, wi].transpose(1,0)))
                else:
                    decoder_input = self.embedding(pred_word)   
                
                if torch.cuda.is_available(): decoder_input = decoder_input.cuda()
        else:
            # Compute pointer-generator output, decoding each (domain, slot) one-by-one
            words_point_out = []
            counter = 0
            for slot in slot_temp:
                hidden = encoded_hidden
                words = []
                slot_emb = slot_emb_dict[slot]
                decoder_input = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)
                for wi in range(max_res_len):
                    dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
                    
                    context_vec, logits, prob = self.attend(encoded_outputs, hidden.squeeze(0), encoded_lens)
                    if wi == 0: 
                        all_gate_outputs[counter] = self.W_gate(context_vec)
                    p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                    p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                    vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                    
                    p_context_ptr = torch.zeros(p_vocab.size())
                    if torch.cuda.is_available(): p_context_ptr = p_context_ptr.cuda()
                    p_context_ptr.scatter_add_(1, story, prob)
                    final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                    vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                    
                    pred_word = torch.argmax(final_p_vocab, dim=1)
                    words.append([self.word_vocab.index2token(w_idx.item()) for w_idx in pred_word])
                    all_point_outputs[counter, :, wi, :] = final_p_vocab
                    
                    if use_teacher_forcing:
                        decoder_input = self.embedding(target_batches[:, counter, wi]) # Chosen word is next input
                    else:
                        decoder_input = self.embedding(pred_word)
                    if torch.cuda.is_available(): decoder_input = decoder_input.cuda()
                counter += 1
                words_point_out.append(words)
        
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


class TRADE(Model):

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

    # def cuda(self):
    #     self.encoder.cuda()
    #     self.decoder.cuda()

        # print("trade params:")
        # for name, param in self.named_parameters():
        #     print(name)

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

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10
        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size, 
            encoded_hidden, encoded_outputs, data['context_len'], story, max_res_len, data['generate_y'], 
            use_teacher_forcing, slot_temp) 
        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def preprocess_data(self, data):
        data["turn_domain"] = [preprocess_domain(item, self.domain_dict) for item in data["turn_domain"]]
        data["generate_y"] = [preprocess_slot(item, self.word_vocab) for item in data["generate_y"]]
        data["context"] = [preprocess(item, self.word_vocab) for item in data["context_plain"]]
        data = collate_fn(data, self.word_vocab, self.gating_dict)
        return data

    def compute_loss(self, data):
        data = self.preprocess_data(data)

        use_teacher_forcing = random.random() < self.args["teacher_forcing_ratio"]
        slot_temp = data["slot_temp"][0]
        (all_point_outputs, gates, 
            words_point_out, words_class_out) = self.encode_and_decode(
                data, use_teacher_forcing, slot_temp)

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            data["generate_y"].contiguous(), #[:,:len(self.point_slots)].contiguous(),
            data["y_lengths"]) #[:,:len(self.point_slots)])
        loss_gate = self.cross_entropy(
            gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)), 
            data["gating_label"].contiguous().view(-1))

        if self.args["use_gate"]:
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr

        return loss, { "LP": loss_ptr, "LG": loss_gate }

    def inference(self, data):
        pass

    def trainable_parameters(self):
        # return filter(lambda p: p.requires_grad, self.parameters())
        return self.parameters()

