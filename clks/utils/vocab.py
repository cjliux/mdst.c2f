#coding: utf-8
import os
import sys

import re
import copy
from collections import Iterable, Iterator
from operator import itemgetter


class Vocabulary():

    def __init__(self, file_name = None):
        self.reset()
        if file_name is not None:
            self.from_file(file_name)

    def reset(self):
        self._vocab = []
        self.unk_index = None
        # self._vocab = ['<unk>']
        # self.unk_index=0
        pass

    def _update_mapping(self):
        self.idx2tok = { i: t for i, t in enumerate(self._vocab) }
        self.tok2idx = { t: i for i, t in enumerate(self._vocab) }

    def extend(self, rhs):
        self._vocab.extend(rhs.get_vocab())
        self._update_mapping()
        return self

    def from_file(self, file_name):
        with open(file_name, "r", encoding="utf8") as fd:
            self._vocab = []
            word = None
            has_empty = False
            for line in fd:
                new_word = line.strip()
                if word is not None:
                    if len(word) > 0:
                        self._vocab.append(word)
                    elif len(new_word) > 0 and not has_empty:
                        self._vocab.append(word)
                        has_empty = True
                word = new_word
            if word is not None and len(word) > 0:
                self._vocab.append(word)
            # self._vocab = [line.strip().split()[0]
            #     for line in fd if len(line.strip()) > 0]
        self._update_mapping()
        return self

    def to_file(self, file_name):
        with open(file_name, "w", encoding='utf8') as fd:
            for w in self._vocab:
                fd.write(w + "\n")
        return self

    def from_tok2idx(self, tok2idx):
        tok_idx = sorted(tok2idx.items(), key=itemgetter(1))
        self._vocab = list(tok for tok, idx in tok_idx)
        self._update_mapping()
        return self

    def from_idx2tok(self, idx2tok):
        idx_tok = sorted(idx2tok.items(), key=itemgetter(0))
        self._vocab = list(tok for idx, tok in idx_tok)
        self._update_mapping()
        return self

    def from_list(self, _vocab):
        self._vocab = _vocab
        self._update_mapping()
        return self

    def has_token(self, token):
        return (token in self.tok2idx)

    def get_vocab(self):
        return copy.deepcopy(self._vocab)

    def __len__(self):
        return len(self._vocab)

    def index2token(self, indices):
        if isinstance(indices, Iterable):
        # if isinstance(indices, Iterable) and not isinstance(tokens, str):
            return [self.index2token(sub_indices) 
                        for sub_indices in indices]
        else:
            return self.idx2tok[indices]

    def token2index(self, tokens, map_func=None):
        if isinstance(tokens, Iterable) and not isinstance(tokens, str):
            return [self.token2index(sub_tokens, map_func) 
                        for sub_tokens in tokens]
        else:
            return self.index_of(tokens, map_func)

    def index_of(self, token, map_func=None):
        if map_func is not None:
            token = map_func(token)
        return self.tok2idx[token] if token in self.tok2idx else self.unk_index
    
    def token_of(self, index):
        return self.idx2tok[index]

    def add_tokens(self, tokens):
        new_tokens = set(tokens).difference(set(self._vocab))
        for token in tokens:
            if token in new_tokens:
                self._vocab.append(token)
                new_tokens.remove(token)
        self._update_mapping()

