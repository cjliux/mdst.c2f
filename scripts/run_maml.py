#coding: utf-8
import os
import sys
import logging

import clks
import utils

from clks.utils.config import Config
import utils.maml as maml
ModeKeys = maml.ModeKeys

try:
    from .__conf__ import *
    from .config_maml import args
except:
    from __conf__ import *
    from config_maml import args


def train(args):
    config = Config().from_file(args["config"])

    if args["dataset"] == "mwoz20":
        from data.mwoz20.interface_maml import build_meta_input_fn
        # (train_loader, dev_loader, test_loader,
        #     test_4d_loader) = prepare_meta_data_loader(
        #         args, args["data_path"], training=True)
        input_fn = build_meta_input_fn(args, args["data_path"], training=True)
        train_task_loader = input_fn(ModeKeys.TRAIN, args, config)

        from models.mwoz20.interface_maml import build_meta_model_fn
        model_fn = build_meta_model_fn(args, config)
        train_spec = model_fn(ModeKeys.TRAIN, train_task_loader)
        eval_spec = model_fn(ModeKeys.EVAL, None)
        scf = maml.MetaLearner().build(args, config, 
            os.path.join(ENV_PATH, "meta_trained"), args["run_id"])
        scf.train(train_spec, eval_spec)
    else:
        raise Exception("Unimplemented")


if __name__=='__main__':
    train(args)
