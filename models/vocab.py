#coding: utf-8
import os
import sys

import re
import copy
from collections import Iterable, Iterator
from operator import itemgetter
from clks.utils.vocab import Vocabulary


UNK_TOKEN = "UNK"
PAD_TOKEN = "PAD"
EOS_TOKEN = "EOS"
SOS_TOKEN = "SOS"


# UNK_TOKEN = "<unk>"
# PAD_TOKEN = "<pad>"
# EOS_TOKEN = "<eos>"
# SOS_TOKEN = "<sos>"


class WordVocab(Vocabulary):

    def reset(self):
        self._vocab = [UNK_TOKEN, PAD_TOKEN, EOS_TOKEN, SOS_TOKEN]
        self.unk_index = 0
        self.pad_index = 1
        self.eos_index = 2
        self.sos_index = 3


class LabelVocab(Vocabulary):

    def __init__(self, file_name=None):
        super().__init__(file_name=file_name)


def add_words_to_vocab(vocab, sent, type):
    add_words_to_vocab_v1(vocab, sent, type)


def add_words_to_vocab_v1(vocab, sent, type):
    new_word_set = set()
    new_word_list = []
    if type == 'utter':
        for word in sent.split(" "):
            if word not in new_word_set:
                new_word_set.add(word)
                new_word_list.append(word)
    elif type == 'slot':
        for slot in sent:
            d, s = slot.split("-")
            if d not in new_word_set:
                new_word_set.add(d)
                new_word_list.append(d)
            for ss in s.split(" "):
                if ss not in new_word_set:
                    new_word_set.add(ss)
                    new_word_list.append(ss)
    elif type == 'belief':
        for slot, value in sent.items():
            d, s = slot.split("-")
            if d not in new_word_set:
                new_word_set.add(d)
                new_word_list.append(d)
            for ss in s.split(" "):
                if ss not in new_word_set:
                    new_word_set.add(ss)
                    new_word_list.append(ss)
            for v in value.split(" "):
                if v not in new_word_set:
                    new_word_set.add(v)
                    new_word_list.append(v)
    vocab.add_tokens(new_word_list)


def add_words_to_vocab_v2(vocab, sent, type):
    new_word_set = set()
    if type == 'utter':
        for word in sent.split(" "):
            word = word.strip()
            if len(word) > 0:
                new_word_set.add(word)
    elif type == 'slot':
        for slot in sent:
            d, s = slot.split("-")
            new_word_set.add(d.strip())
            for ss in s.split(" "):
                ss = ss.strip()
                if len(ss) > 0:
                    new_word_set.add(ss)
    elif type == 'belief':
        for slot, value in sent.items():
            d, s = slot.split("-")
            new_word_set.add(d.strip())
            for ss in s.split(" "):
                ss = ss.strip()
                if len(ss) > 0:
                    new_word_set.add(ss)
            for v in value.split(" "):
                v = v.strip()
                if len(v) > 0:
                    new_word_set.add(v)
    vocab.add_tokens(new_word_set)

