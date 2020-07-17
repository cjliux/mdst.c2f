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
from collections import defaultdict, OrderedDict
from data.fix_label import fix_general_label_error


EXPERIMENT_DOMAINS = [
    "hotel", "train", "restaurant", "attraction", "taxi"]


# def get_slot_information(ontology):
#     ontology_domains = dict([
#         (k, v) for k, v in ontology.items() 
#             if k.split("-")[0] in EXPERIMENT_DOMAINS])
#     slots = [k.replace(" ","").lower() 
#         if ("book" not in k) else k.lower() 
#             for k in ontology_domains.keys()]
#     return slots


def read_slot_temp():
    with open('./analysis/all_slots.txt', 'r', encoding='utf8') as fd:
        all_slots = [l.strip() for l in fd if len(l.strip()) > 0]
    return all_slots


def read_dial_data(data_file, all_slots, max_line=None):
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = defaultdict(int)

    with open(data_file, 'r', encoding='utf8') as fd:
        dials = json.load(fd)
    
    allow_domains = set(EXPERIMENT_DOMAINS)
    allow_slots = copy.deepcopy(all_slots)

    cnt_lin = 1
    for dial_dict in dials:
        dialog_history = ""
        last_belief_dict = {}
        # filter and count domains
        for domain in dial_dict["domains"]:
            if domain not in EXPERIMENT_DOMAINS:
                continue
            domain_counter[domain] += 1

        last_dict = fix_general_label_error(dial_dict["dialogue"][-1]["belief_state"], False, all_slots)
        turn_belief_dict = OrderedDict([(k, v) 
                for k, v in last_dict.items() if k in allow_slots])
        if len(turn_belief_dict) == 0:
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
                for k, v in turn_belief_dict.items() if k in allow_slots])
            turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]

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
                "turn_belief_dict": turn_belief_dict,
                "turn_belief_list": turn_belief_list,
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


def get_value_set(dial_data, all_slots):
    value_dist = {}
    for slot in all_slots:
        value_dist[slot] = defaultdict(int)

    for data_detail in dial_data:
        turn_belief_dict = data_detail['turn_belief_dict']
        for k, v in turn_belief_dict.items():
            value_dist[k][v] += 1

    value_set = {}
    for slot in all_slots:
        value_set[slot] = set(value_dist[slot].keys())
    return value_set


def stats_unseen_value(args, all_slots):
    train_dials, _, _ = read_dial_data(
        data_file=os.path.join(args["data_path"], "proc/train_dials.json"),
        all_slots=all_slots)
    dev_dials, _, _ = read_dial_data(
        data_file=os.path.join(args["data_path"], "proc/dev_dials.json"),
        all_slots=all_slots)
    test_dials, _, _ = read_dial_data(
        data_file=os.path.join(args["data_path"], "proc/test_dials.json"),
        all_slots=all_slots)

    train_value_set = get_value_set(train_dials, all_slots)
    dev_value_set = get_value_set(dev_dials, all_slots
    )
    test_value_set = get_value_set(test_dials, all_slots)
    unseen_value_set = {}
    for slot in all_slots:
        unseen = dev_value_set[slot].union(test_value_set[slot])
        # unseen = test_value_set[slot]
        unseen_value_set[slot] = unseen.difference(train_value_set[slot])

    for slot in all_slots:
        with open('./analysis/unseen_{}.txt'.format(slot), 'w', encoding='utf8') as fd:
            for v in unseen_value_set[slot]:
                fd.write(v + '\n')


def read_unseen_value(all_slots):
    unseen_value_set = {}
    for slot in all_slots:
        with open('./analysis/unseen_{}.txt'.format(slot), 'r', encoding='utf8') as fd:
            unseen_value_set[slot] = set([v.strip() for v in  fd if len(v.strip()) > 0])
    return unseen_value_set


def stats_none(anl_dict, all_slots, unseen_value_set):
    total_cnt, error_cnt = defaultdict(int), defaultdict(int)

    error_dc2val = {sl:defaultdict(int) for sl in all_slots}
    error_none2val = {sl:defaultdict(int) for sl in all_slots}
    error_val2dc = {sl:defaultdict(int) for sl in all_slots}
    error_val2none = {sl:defaultdict(int) for sl in all_slots}
    error_none2dc = {sl:defaultdict(int) for sl in all_slots}
    error_dc2none = {sl:defaultdict(int) for sl in all_slots}
    
    corr_none = {sl:defaultdict(int) for sl in all_slots}
    corr_dc = {sl:defaultdict(int) for sl in all_slots}
    corr_seen_value = {sl:defaultdict(int) for sl in all_slots}
    corr_unseen_value = {sl:defaultdict(int) for sl in all_slots}

    for dial_id, dial in anl_dict.items():
        for turn_id, turn in dial.items():
            infer = turn['infer']

            for slot in all_slots:
                tgt = turn[slot]['tgt']
                
                val_type = 0
                final = None
                gate = infer[slot]['gate']
                if gate == 'none':
                    final = 'none'
                elif gate == 'ptr':
                    if infer[slot]['rank'][0][0] != 'none':
                        final = infer[slot]['rank'][0][0]
                        val_type = 1
                    else:
                        final = infer[slot]['gen']
                        val_type = 2
                else:
                    final = 'dontcare'

                total_cnt[slot] += 1
                if final == tgt:
                    if final == 'none':
                        corr_none[slot][val_type] += 1
                    elif final == 'dontcare':
                        corr_dc[slot][val_type] += 1
                    else: 
                        if final in unseen_value_set[slot]:
                            corr_unseen_value[slot][val_type] += 1
                        else:
                            corr_seen_value[slot][val_type] += 1
                else:
                    error_cnt[slot] += 1

                    if tgt == 'none':
                        if final == 'dontcare':
                            error_none2dc[slot][val_type] += 1
                        else:
                            error_none2val[slot][val_type] += 1
                    elif tgt == 'dontcare':
                        if final == 'none':
                            error_dc2none[slot][val_type] += 1
                        else:
                            error_dc2val[slot][val_type] += 1
                    else:
                        if final == 'none':
                            error_val2none[slot][val_type] += 1
                        else:
                            error_val2dc[slot][val_type] += 1

    print('gate rank ptr')
    print('none2val: {}'.format(error_none2val))
    print('dc2val: {}'.format(error_dc2val))
    print('val2none: {}'.format(error_val2none))
    print('val2dc: {}'.format(error_val2dc))
    print('none2dc: {}'.format(error_none2dc))
    print('dc2none: {}'.format(error_dc2none))
    print('none: {}'.format(corr_none))
    print('dc: {}'.format(corr_dc))
    print('seen: {}'.format(corr_seen_value))
    print('unseen: {}'.format(corr_unseen_value))


def stats_none_v2(anl_dict, all_slots, unseen_value_set):
    total_cnt, error_cnt = defaultdict(int), defaultdict(int)

    error_dc2val = defaultdict(int)
    error_none2val = defaultdict(int)
    error_val2dc = defaultdict(int)
    error_val2none = defaultdict(int)
    error_none2dc = defaultdict(int)
    error_dc2none = defaultdict(int)
    
    corr_none = defaultdict(int)
    corr_dc = defaultdict(int)
    corr_seen_value = defaultdict(int)
    corr_unseen_value = defaultdict(int)

    for dial_id, dial in anl_dict.items():
        for turn_id, turn in dial.items():
            infer = turn['infer']

            for slot in all_slots:
                tgt = turn[slot]['tgt']
                
                val_type = 0
                final = None
                gate = infer[slot]['gate']
                if gate == 'none':
                    final = 'none'
                elif gate == 'ptr':
                    if infer[slot]['rank'][0][0] != 'none':
                        final = infer[slot]['rank'][0][0]
                        val_type = 1
                    else:
                        final = infer[slot]['gen']
                        val_type = 2
                else:
                    final = 'dontcare'

                total_cnt[slot] += 1
                if final == tgt:
                    if final == 'none':
                        corr_none[val_type] += 1
                    elif final == 'dontcare':
                        corr_dc[val_type] += 1
                    else: 
                        if final in unseen_value_set[slot]:
                            corr_unseen_value[val_type] += 1
                        else:
                            corr_seen_value[val_type] += 1
                else:
                    error_cnt[slot] += 1

                    if tgt == 'none':
                        if final == 'dontcare':
                            error_none2dc[val_type] += 1
                        else:
                            error_none2val[val_type] += 1
                    elif tgt == 'dontcare':
                        if final == 'none':
                            error_dc2none[val_type] += 1
                        else:
                            error_dc2val[val_type] += 1
                    else:
                        if final == 'none':
                            error_val2none[val_type] += 1
                        else:
                            error_val2dc[val_type] += 1

    print('gate rank ptr')
    print('none2val: {}'.format(error_none2val))
    print('dc2val: {}'.format(error_dc2val))
    print('val2none: {}'.format(error_val2none))
    print('val2dc: {}'.format(error_val2dc))
    print('none2dc: {}'.format(error_none2dc))
    print('dc2none: {}'.format(error_dc2none))
    print('none: {}'.format(corr_none))
    print('dc: {}'.format(corr_dc))
    print('seen: {}'.format(corr_seen_value))
    print('unseen: {}'.format(corr_unseen_value))


def stats_example():
    pass


def main(args):
    all_slots = read_slot_temp()
    # stats_unseen_value(args, all_slots)
    unseen_value_set = read_unseen_value(all_slots)

    with open(args["anl_file"], 'r', encoding='utf8') as fd:
        anl_dict = json.load(fd)
    
    stats_none_v2(anl_dict, all_slots, unseen_value_set)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="../data/woz/mwoz21")
    parser.add_argument("--anl_file", type=str, required=False)

    return vars(parser.parse_args())


if __name__=='__main__':
    args = parse_args()

    args['anl_file'] = "../nolol_mwoz21_trained/c2fa2o_2--bsz32/analyze_c2fa2o_2--bsz32_TEST.json"
    main(args)


