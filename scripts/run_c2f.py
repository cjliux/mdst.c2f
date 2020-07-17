#coding: utf-8
import os
import sys
import logging

try:
    from .__conf__ import *
    from .config import args
except:
    from __conf__ import *
    from config import args
sys.path.append(MDST_HOME)

import clks
import utils

from clks.utils.config import Config
import utils.scaffold as scaffold
ModeKeys = scaffold.ModeKeys


def train(args):
    config = Config().from_file(args["config"])

    from data.interface import prepare_data_loader
    (train_loader, dev_loader, test_loader,
        test_4d_loader) = prepare_data_loader(
            args, args["data_path"], training=True)
    from models.interface_c2f import build_model_fn
    model_fn = build_model_fn(args, config)
    train_spec = model_fn(ModeKeys.TRAIN, train_loader)
    eval_spec = model_fn(ModeKeys.EVAL, dev_loader)
    scf = scaffold.Scaffold().build(
        args, config, os.path.join(ENV_PATH, 
        "nolol_{}_trained".format(args["dataset"])), args["run_id"])
    scf.train(train_spec, eval_spec)


def evaluate(args):
    config = Config().from_file(args["config"])

    from data.interface import prepare_data_loader
    (train_loader, dev_loader, test_loader,
        test_4d_loader) = prepare_data_loader(
            args, args["data_path"], training=False)
    from models.interface_c2f import build_model_fn
    model_fn = build_model_fn(args, config)
    scf = scaffold.Scaffold().build(args, config,
                os.path.join(ENV_PATH, 
                "nolol_{}_trained".format(args["dataset"])), args["run_id"])

    # for dev set
    if args["run_dev_testing"]:
        print("Development Set ...")
        eval_spec = model_fn(ModeKeys.EVAL, dev_loader)
        scf.evaluate(eval_spec)

    if args['except_domain'] != "" and args["run_except_4d"]:
        print("Test Set on 4 domains...")
        eval_spec = model_fn(ModeKeys.EVAL, test_4d_loader)
        scf.evaluate(eval_spec)

    # for test set
    print("Test Set ...")
    eval_spec = model_fn(ModeKeys.TEST, test_loader)
    scf.evaluate(eval_spec)


def test(args):
    raise Exception("Unimplemented")


if __name__ == '__main__':
    os.chdir(MDST_HOME)
    # import ipdb
    # with ipdb.launch_ipdb_on_exception():
    {"train": train, "eval": evaluate, "test": test}[args["mode"]](args)
