#coding: utf-8
import os
import sys
import logging

try:
    from .__conf__ import *
    from .config_meta_rl2 import args
except:
    from __conf__ import *
    from config_meta_rl2 import args
sys.path.append(MDST_HOME)

import clks
import utils

from clks.utils.config import Config
import utils.meta_rl2 as meta_rl2
ModeKeys = meta_rl2.ModeKeys


def train(args):
    config = Config().from_file(args["config"])

    if args["dataset"] == "mwoz20":
        from data.mwoz20.interface_meta_rl3 import prepare_data_loader_4_meta_rl
        (train_loader, dev_loader, test_loader,
            test_4d_loader) = prepare_data_loader_4_meta_rl(
                args, args["data_path"], training=True)
        from models.mwoz20.interface_meta_rl2 import build_model_fn_4_meta_rl
        model_fn = build_model_fn_4_meta_rl(args, config)
        train_spec = model_fn(ModeKeys.TRAIN, train_loader)
        eval_spec = model_fn(ModeKeys.EVAL, dev_loader)
        scf = meta_rl2.MetaRLLearner().build(
            args, config, os.path.join(ENV_PATH, "meta_rl3_trained"), args["run_id"])
        scf.train(train_spec, eval_spec)
    else:
        raise Exception("Unimplemented")


def evaluate(args):
    config = Config().from_file(args["config"])

    if args["dataset"] == "mwoz20":
        from data.mwoz20.interface_meta_rl3 import prepare_data_loader_4_meta_rl
        (train_loader, dev_loader, test_loader,
            test_4d_loader) = prepare_data_loader_4_meta_rl(
                args, args["data_path"], training=False)
        from models.mwoz20.interface_meta_rl2 import build_model_fn_4_meta_rl
        model_fn = build_model_fn_4_meta_rl(args, config)
        # for dev set
        if args["run_dev_testing"]:
            print("Development Set ...")
            eval_spec = model_fn(ModeKeys.EVAL, dev_loader)
            scf = meta_rl2.MetaRLLearner().build(args, config,
                        os.path.join(ENV_PATH, "meta_rl3_trained"), args["run_id"])
            scf.evaluate(eval_spec)

        if args['except_domain'] != "" and args["run_except_4d"]:
            print("Test Set on 4 domains...")
            eval_spec = model_fn(ModeKeys.EVAL, test_4d_loader)
            scf = meta_rl2.MetaRLLearner().build(args, config,
                        os.path.join(ENV_PATH, "meta_rl3_trained"), args["run_id"])
            scf.evaluate(eval_spec)

        # for test set
        print("Test Set ...")
        eval_spec = model_fn(ModeKeys.TEST, test_loader)
        scf = meta_rl2.MetaRLLearner().build(args, config,
                    os.path.join(ENV_PATH, "meta_rl3_trained"), args["run_id"])
        scf.evaluate(eval_spec)
    else:
        raise Exception("Unimplemented")


def test(args):
    raise Exception("Unimplemented")


if __name__ == '__main__':
    os.chdir(MDST_HOME)
    {"train": train, "eval": evaluate, "test": test}[args["mode"]](args)
