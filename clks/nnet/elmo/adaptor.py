#coding: utf-8
import os
import sys
sys.path.append("../../..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
import clks.func.tensor as T

# from .__conf__ import CACHE_DIR


# options_file = os.path.join(CACHE_DIR, 
#     "elmo_2x4096_512_2048cnn_2xhighway_options.json")
# weight_file = os.path.join(CACHE_DIR, 
#     "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")

class ElmoAdaptor(nn.Module):

    def __init__(self, options_file, weight_file, **elmo_args):
        """
        Args:
            config: 
                elmo_args
        """
        super().__init__()
        self.options_file = options_file
        self.weight_file = weight_file
        self.elmo = Elmo(options_file, weight_file, **elmo_args)

    def process_sent(self, tokens):
        """
        Args:
            tokens: list of str tokens. (in batch format, 2-level)
        Returns:
            x, m: seq-first.
        """ 
        token_ids = batch_to_ids(tokens)
        return T.to_cuda(token_ids)

    def __call__(self, input):
        """
        Args:
            input: list of token ids after process_sent (or 
                    lower level batch_to_ids)
        Returns:
            representations: ?, batch_first
        """
        result_dict = self.elmo(input)
        representations = result_dict["elmo_representations"]
        if self.debug_level > 0:
            self.logger.info("representations: {}".format(type(representations)))
        return representations
