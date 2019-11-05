#coding: utf-8
import os
import sys
import json
import copy
import pickle
import random
from tqdm import tqdm
from embeddings import GloveEmbedding, KazumaCharEmbedding
from collections import defaultdict, OrderedDict
try:
    from .vocab import WordVocab, LabelVocab, add_words_to_vocab
except:
    from vocab import WordVocab, LabelVocab, add_words_to_vocab

from data.mwoz20.interface import read_ontology
from data.mwoz20.fix_label import fix_general_label_error


EXPERIMENT_DOMAINS = [
    "hotel", "train", "restaurant", "attraction", "taxi"]


def walk_dials(args, data_file, split, all_slots, 
                        word_vocab, mem_vocab, max_line=None):
    max_resp_len, max_value_len = 0, 0
    domain_counter = defaultdict(int)

    with open(data_file, 'r', encoding='utf8') as fd:
        dials = json.load(fd)

    # determine training data ratio, default is 100%
    if split=="train" and args["data_ratio"] < 1:
        random.Random(10).shuffle(dials)
        dials = dials[:int(len(dials)*float(args['data_ratio']))]

    if args["all_vocab"] or split == "train":
        for dial_dict in dials:
            for ti, turn in enumerate(dial_dict["dialogue"]):
                add_words_to_vocab(word_vocab, turn["system_transcript"], "utter")
                add_words_to_vocab(word_vocab, turn["transcript"], "utter")

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
            if d in args["only_domain"] for args["only_domain"] in d])
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

        # Unseen domain setting
        if args["only_domain"] != "" and args["only_domain"] not in dial_dict["domains"]:
            continue
        if (args["except_domain"] != "" and split == "test" and args["except_domain"] not in dial_dict["domains"]) or \
            (args["except_domain"] != "" and split != "test" and [args["except_domain"]] == dial_dict["domains"]): 
            continue

        # read data
        for ti, turn in enumerate(dial_dict["dialogue"]):
            turn_domain = turn["domain"]
            turn_id = turn["turn_idx"]
            turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
            turn_uttr_strip = turn_uttr.strip()
            dialog_history +=  (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
            source_text = dialog_history.strip()
            turn_belief_dict = fix_general_label_error(turn["belief_state"], False, all_slots)
            turn_belief_dict = OrderedDict([(k, v) 
                for k, v in turn_belief_dict.items() if k.split('-')[0] in allow_domains])
            turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]

            # vocab
            if args["all_vocab"] or split == "train":
                add_words_to_vocab(mem_vocab, turn_belief_dict, 'belief')

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

                    if max_value_len < len(turn_belief_dict[slot]):
                        max_value_len = len(turn_belief_dict[slot])
                else:
                    generate_y.append("none")
                    gating_label.append("none")

            max_resp_len = max(max_resp_len, len(source_text.split()))

        cnt_lin += 1
        if max_line and cnt_lin >= max_line:
            break

    if "t{}".format(max_value_len-1) not in mem_vocab.get_vocab():
        for time_i in range(max_value_len):
            add_words_to_vocab(mem_vocab, "t{}".format(time_i), 'utter')

    print("domain_counter", domain_counter)
    return word_vocab, mem_vocab, allow_domains, allow_slots


def build_vocab(args):
    all_slots = read_ontology(args["data_path"])

    word_vocab, mem_vocab = WordVocab(), WordVocab()
    add_words_to_vocab(word_vocab, all_slots, 'slot')
    add_words_to_vocab(mem_vocab, all_slots, 'slot')
    
    slots_list = [all_slots]
    train_file = os.path.join(args["data_path"], "proc", "train_dials.json")
    word_vocab, mem_vocab, allow_domains, allow_slots = walk_dials(
        args, train_file, "train", all_slots, word_vocab, mem_vocab)
    slots_list.append(allow_slots)
    dev_file = os.path.join(args["data_path"], "proc", "dev_dials.json")
    word_vocab, mem_vocab, allow_domains, allow_slots = walk_dials(
        args, dev_file, "dev", all_slots, word_vocab, mem_vocab)
    slots_list.append(allow_slots)
    test_file = os.path.join(args["data_path"], "proc", "test_dials.json")
    word_vocab, mem_vocab, allow_domains, allow_slots = walk_dials(
        args, test_file, "test", all_slots, word_vocab, mem_vocab)
    slots_list.append(allow_slots)

    return word_vocab, mem_vocab, slots_list


def dump_pretrained_emb(vocab, dump_path):
    print("Dumping pretrained embeddings...")
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for i in tqdm(range(len(vocab.get_vocab()))):
        w = vocab.index2token(i)
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-dp", "--data_path", default="../data/woz/mwoz20/")
    parser.add_argument("-avoc", "--all_vocab", action='store_true')
    parser.add_argument("--data_ratio", default=1, type=float)
    parser.add_argument('-exceptd','--except_domain', help='', required=False, default="", type=str)
    parser.add_argument('-onlyd','--only_domain', help='', required=False, default="", type=str)
    parser.add_argument('-pemb', '--emb_path', help='pretrained embeddings', 
        type=str, default="../trade-dst/embeddings")

    return vars(parser.parse_args())


if __name__=='__main__':
    args = parse_args()
    save_path = os.path.join(args["data_path"], "save")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # build vocab
    word_vocab, mem_vocab, slots_list = build_vocab(args)
    word_vocab_file = os.path.join(save_path, "word_vocab.txt")
    word_vocab.to_file(word_vocab_file)
    mem_vocab_file = os.path.join(save_path, "mem_vocab.txt")
    mem_vocab.to_file(mem_vocab_file)
    print(len(word_vocab), len(mem_vocab))
    
    for i, name in enumerate(
            ["all_slots", "slots_train", "slots_dev", "slots_test"]):
        slots_vocab_file = os.path.join(save_path, "{}.txt".format(name))
        LabelVocab().from_list(slots_list[i]).to_file(slots_vocab_file)
    domain_dict = {"attraction":0, "restaurant":1, "taxi":2, 
        "train":3, "hotel":4, "hospital":5, "bus":6, "police":7}
    domain_dict_file = os.path.join(save_path, "domain_dict.txt")
    LabelVocab().from_tok2idx(domain_dict).to_file(domain_dict_file)
    gating_dict = {"ptr":0, "dontcare":1, "none":2}
    gating_dict_file = os.path.join(save_path, "gating_dict.txt")
    LabelVocab().from_tok2idx(gating_dict).to_file(gating_dict_file)

    os.environ['HOME'] = './'
    os.environ['EMBEDDINGS_ROOT'] = args["emb_path"]

    # dump embeddings
    emb_dump_file = os.path.join(
        save_path, "emb{}.json".format(len(word_vocab)))
    dump_pretrained_emb(word_vocab, emb_dump_file)
