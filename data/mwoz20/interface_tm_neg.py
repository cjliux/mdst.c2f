#coding: utf-8
import os
import sys
import copy
import json
import random
import torch
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict

try:
    from .fix_label import fix_general_label_error
except: 
    from fix_label import fix_general_label_error

import utils.scaffold
ModeKeys = utils.scaffold.ModeKeys


EXPERIMENT_DOMAINS = [
    "hotel", "train", "restaurant", "attraction", "taxi"]


def get_slot_information(ontology):
    ontology_domains = dict([
        (k, v) for k, v in ontology.items() 
            if k.split("-")[0] in EXPERIMENT_DOMAINS])
    slots = [k.replace(" ","").lower() 
        if ("book" not in k) else k.lower() 
            for k in ontology_domains.keys()]
    return slots


def read_ontology(data_path):
    ont_file = os.path.join(data_path, "ontology.json")
    with open(ont_file, 'r', encoding='utf8') as fd:
        ontology = json.load(fd)
    all_slots = get_slot_information(ontology)
    return all_slots


def read_dial_data(args, data_file, split, all_slots, training, max_line=None):
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = defaultdict(int)

    # data_file = os.path.join(data_path, "proc", "{}_dials.json".format(split))
    with open(data_file, 'r', encoding='utf8') as fd:
        dials = json.load(fd)

    # determine training data ratio, default is 100%
    if training and split=="train" and args["data_ratio"] < 1:
        random.Random(10).shuffle(dials)
        dials = dials[:int(len(dials)*float(args['data_ratio']))]

    allow_domains = set(EXPERIMENT_DOMAINS)
    allow_slots = copy.deepcopy(all_slots)
    if args["except_domain"] != "":
        if split in ["train", "dev"]:
            allow_domains = set([d for d in allow_domains 
                if args["except_domain"] not in d and d not in args["except_domain"]])
        else:
            allow_domains = set([d for d in allow_domains 
                if args["except_domain"] in d and d in args["except_domain"]])
    elif args["only_domain"] != "":
        allow_domains = set([d for d in allow_domains 
            if d in args["only_domain"] or args["only_domain"] in d])
    allow_slots = [k for k in allow_slots if k.split('-')[0] in allow_domains]

    cnt_lin = 1
    for dial_dict in dials:
        dialog_history = ""
        last_belief_dict = {}
        # filter and count domains
        for domain in dial_dict["domains"]:
            if domain not in EXPERIMENT_DOMAINS:
                continue
            domain_counter[domain] += 1

        # Unseen domain setting: skip dial
        if args["only_domain"] != "" and args["only_domain"] not in dial_dict["domains"]:
            continue
        if (args["except_domain"] != "" and split == "test" and args["except_domain"] not in dial_dict["domains"]) or \
            (args["except_domain"] != "" and split != "test" and [args["except_domain"]] == dial_dict["domains"]): 
            continue

        # read data
        for ti, turn in enumerate(dial_dict["dialogue"]):
            turn_domain = turn["domain"]
            turn_id = turn["turn_idx"]
            
            turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"] + " ; "
            turn_uttr_strip = turn_uttr.strip()

            dialog_history +=  (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
            source_text = dialog_history.strip()

            turn_belief_dict = fix_general_label_error(turn["belief_state"], False, all_slots)
            turn_belief_dict = OrderedDict([(k, v) 
                for k, v in turn_belief_dict.items() if k.split('-')[0] in allow_domains])
            turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]

            turn_label_dict = fix_general_label_error(turn["turn_label"], True, all_slots)
            turn_label_dict = OrderedDict([(k, v)
                for k, v in turn_label_dict.items() if k.split('-')[0] in allow_domains])
            turn_label_list = [str(k)+'-'+str(v) for k, v in turn_label_dict.items()]

            # if (args["all_vocab"] or split=="train") and training:
            #     mem_lang.index_words(turn_belief_dict, 'belief')

            class_label, generate_y, slot_mask, gating_label  = [], [], [], []
            start_ptr_label, end_ptr_label = [], []
            for slot in allow_slots:
                if slot in turn_belief_dict.keys(): 
                    generate_y.append(turn_belief_dict[slot])

                    if turn_belief_dict[slot] == "dontcare":
                        gating_label.append("dontcare")
                    elif turn_belief_dict[slot] == "none":
                        gating_label.append("none")
                    else:
                        gating_label.append("ptr")

                    max_value_len = max(max_value_len, len(turn_belief_dict[slot]))
                else:
                    generate_y.append("none")
                    gating_label.append("none")

            data_detail = {
                "ID": dial_dict["dialogue_idx"],
                "domains": dial_dict["domains"],
                "turn_domain": turn_domain,
                "turn_id": turn_id,
                "dialog_history": source_text,
                "turn_belief": turn_belief_list,
                "turn_label": turn_label_list,
                "gating_label": gating_label,
                "turn_uttr": turn_uttr_strip,
                "generate_y": generate_y
            }
            data.append(data_detail)

            max_resp_len = max(max_resp_len, len(source_text.split()))

        cnt_lin += 1
        if max_line and cnt_lin >= max_line:
            break

    # if "t{}".format(max_value_len-1) not in mem_lang.word2index.keys() and training:
    #     for time_i in range(max_value_len):
    #         mem_lang.index_words("t{}".format(time_i), 'utter') 

    print("domain_counter", domain_counter)
    return data, max_resp_len, allow_slots


class NegPool:

    def __init__(self, args, data_info, slot_temp):
        super().__init__()
        self.n_negs = args["num_negs"]
        self.turn_id = data_info["turn_id"]
        self.turn_belief = data_info["turn_belief"]
        self.turn_label = data_info["turn_label"]
        self.turn_uttr = data_info["turn_uttr"]
        self.slot_temp = slot_temp
        self.build_index()

    def build_index(self):
        self.neg_pool = {}
        for itid, tid in enumerate(self.turn_id):
            turn_label = self.turn_label[itid]
            turn_uttr = self.turn_uttr[itid]
            
            turn_slots = set([lbl[:lbl.rindex('-')] for lbl in turn_label])
            for slot in self.slot_temp:
                if slot not in turn_slots:
                    if slot not in self.neg_pool:
                        self.neg_pool[slot] = set()
                    self.neg_pool[slot].add(itid)
            if len(turn_slots) != 0:
                if "_empty_" not in self.neg_pool:
                    self.neg_pool["_empty_"] = set()
                self.neg_pool["_empty_"].add(itid)
        self.cache = {}

    def get_negs(self, index):
        """
        spec: [[uttr x n_negs] x n_keys]
        """
        keys = self.turn_label[index]
        negs = []
        if len(keys) == 0:
            pool = self.neg_pool["_empty_"]
        else:
            if tuple(keys) in self.cache:
                pool = self.cache[tuple(keys)]
            else:
                key = keys[0]
                pool = self.neg_pool[key[:key.rindex('-')]]
                for key in keys[1:]:
                    slot = key[:key.rindex('-')]
                    pool = pool.intersection(self.neg_pool[slot])
                self.cache[tuple(keys)] = pool
        neg_inds = np.random.permutation(list(pool))[:self.n_negs].tolist()
        negs = [self.turn_uttr[i] for i in neg_inds]
        return negs


class MultiWOZ(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, args, data_info, slot_temp):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_domain = data_info['turn_domain']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.turn_belief = data_info['turn_belief']
        self.turn_label = data_info['turn_label']
        self.gating_label = data_info['gating_label']
        self.turn_uttr = data_info['turn_uttr']
        self.generate_y = data_info["generate_y"]
        self.num_total_seqs = len(self.dialog_history)
        self.slot_temp = slot_temp
        self.neg_pool = NegPool(args, data_info, slot_temp)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        negs = self.neg_pool.get_negs(index)
        item_info = {
            "ID": self.ID[index], 
            "turn_id": self.turn_id[index], 
            "turn_belief": self.turn_belief[index], 
            "gating_label": self.gating_label[index], 
            "context_plain":self.dialog_history[index].split(), 
            "turn_uttr_plain": self.turn_uttr[index], 
            "turn_domain": self.turn_domain[index], 
            "generate_y": [v.split() for v in self.generate_y[index]],
            "slot_temp": self.slot_temp,
            "negs_plain": negs
        }
        return item_info

    def __len__(self):
        return self.num_total_seqs


def collate_fn(data):
    data.sort(key=lambda x: len(x['context_plain']), reverse=True) 
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    return item_info


def sample_map_and_batch(args, pairs, batch_size, shuffle, slot_temp):
    if(shuffle and args['fisher_sample']>0):
        random.shuffle(pairs)
        pairs = pairs[:args['fisher_sample']]

    # distribute
    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []
    # distribute
    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k]) 

    dataset = MultiWOZ(args, data_info, slot_temp)

    if args["imbalance_sampler"] and shuffle:
        data_loader = torch.utils.data.DataLoader(
                                dataset=dataset,
                                batch_size=batch_size,
                                # shuffle=shuffle,
                                sampler=ImbalancedDatasetSampler(dataset),
                                collate_fn=collate_fn)
    else:
        data_loader = torch.utils.data.DataLoader(
                                dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=collate_fn)
    return data_loader


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] 
                                                for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.turn_domain[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def prepare_data_loader(args, data_path, training):
    batch_size = args['batch'] if args['batch'] else 100
    # eval_batch = args["eval_batch"] if args["eval_batch"] else 100
    eval_batch = batch_size

    # Create saving folder
    folder_name = os.path.join(data_path, "save")
    print("folder_name", folder_name)
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)

    all_slots = read_ontology(data_path)

    # gating_dict = {"ptr":0, "dontcare":1, "none":2}

    if training:
        train_file = os.path.join(data_path, "proc", "train_dials.json")
        pair_train, train_max_len, slot_train = read_dial_data(
            args, train_file, "train", all_slots, training)
        train_loader = sample_map_and_batch(args, pair_train, batch_size, True, slot_train)
        # train_loader = sample_map_and_batch(args, pair_train, batch_size, False)

        dev_file = os.path.join(data_path, "proc", "dev_dials.json")
        pair_dev, dev_max_len, slot_dev = read_dial_data(
            args, dev_file, "dev", all_slots, training)
        dev_loader   = sample_map_and_batch(args, pair_dev, eval_batch, False, slot_dev)

        test_file = os.path.join(data_path, "proc", "test_dials.json")
        pair_test, test_max_len, slot_test = read_dial_data(
            args, test_file, "test", all_slots, training)
        test_loader  = sample_map_and_batch(args, pair_test, eval_batch, False, slot_test)
    else:
        pair_train, train_max_len, slot_train = [], 0, {}
        train_loader = []

        dev_file = os.path.join(data_path, "proc", "dev_dials.json")
        pair_dev, dev_max_len, slot_dev = read_dial_data(
            args, dev_file, "dev", all_slots, training)
        dev_loader   = sample_map_and_batch(args, pair_dev, eval_batch, False, slot_dev)

        test_file = os.path.join(data_path, "proc", "test_dials.json")
        pair_test, test_max_len, slot_test = read_dial_data(
            args, test_file, "test", all_slots, training)
        test_loader  = sample_map_and_batch(args, pair_test, eval_batch, False, slot_test)

    test_4d_loader = []
    if args['except_domain']!="":
        test_file = os.path.join(data_path, "proc", "test_dials.json")
        pair_test_4d, _, slot_test_4d = read_dial_data(
            args, test_file, "dev", all_slots, training)
        test_4d_loader  = sample_map_and_batch(args, pair_test_4d, eval_batch, False, slot_dev)

    max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))  
    print("Max. length of dialog words for RNN: %s " % max_word)

    slots_list = [all_slots, slot_train, slot_dev, slot_test]
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(slots_list[2]))))
    print(slots_list[2])
    print("[Test Set Slots]: Number is {} in total".format(str(len(slots_list[3]))))
    print(slots_list[3])
    return train_loader, dev_loader, test_loader, test_4d_loader


if __name__=='__main__':
    pass