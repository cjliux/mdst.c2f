#coding: utf-8
import os
import sys
import logging

try:
    from .__conf__ import *
    from .config_tm_neg import args
except:
    from __conf__ import *
    from config_tm_neg import args
sys.path.append(MDST_HOME)

import clks
import utils

from clks.utils.config import Config
import utils.scaffold as scaffold
ModeKeys = scaffold.ModeKeys



def train(args):
    config = Config().from_file(args["config"])

    if args["dataset"] == "mwoz20":
        from data.mwoz20.interface_tm_neg import prepare_data_loader
        (train_loader, dev_loader, test_loader,
            test_4d_loader) = prepare_data_loader(
                args, args["data_path"], training=True)
        from models.mwoz20.interface import build_model_fn
        model_fn = build_model_fn(args, config)
        train_spec = model_fn(ModeKeys.TRAIN, train_loader)
        eval_spec = model_fn(ModeKeys.EVAL, dev_loader)
        scf = scaffold.Scaffold().build(
            args, config, os.path.join(ENV_PATH, "trained"), args["run_id"])
        scf.train(train_spec, eval_spec)
    else:
        raise Exception("Unimplemented")


def evaluate(args):
    config = Config().from_file(args["config"])

    if args["dataset"] == "mwoz20":
        from data.mwoz20.interface_tm_neg import prepare_data_loader
        (train_loader, dev_loader, test_loader,
            test_4d_loader) = prepare_data_loader(
                args, args["data_path"], training=False)
        from models.mwoz20.interface import build_model_fn
        model_fn = build_model_fn(args, config)
        # for dev set
        if args["run_dev_testing"]:
            print("Development Set ...")
            eval_spec = model_fn(ModeKeys.EVAL, dev_loader)
            scf = scaffold.Scaffold().build(args, config,
                        os.path.join(ENV_PATH, "trained"), args["run_id"])
            scf.evaluate(eval_spec)

        if args['except_domain'] != "" and args["run_except_4d"]:
            print("Test Set on 4 domains...")
            eval_spec = model_fn(ModeKeys.EVAL, test_4d_loader)
            scf = scaffold.Scaffold().build(args, config,
                        os.path.join(ENV_PATH, "trained"), args["run_id"])
            scf.evaluate(eval_spec)

        # for test set
        print("Test Set ...")
        eval_spec = model_fn(ModeKeys.TEST, test_loader)
        scf = scaffold.Scaffold().build(args, config,
                                        os.path.join(ENV_PATH, "trained"), args["run_id"])
        scf.evaluate(eval_spec)
    else:
        raise Exception("Unimplemented")


def test(args):
    raise Exception("Unimplemented")


if __name__ == '__main__':
    os.chdir(MDST_HOME)
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        {"train": train, "eval": evaluate, "test": test}[args["mode"]](args)
