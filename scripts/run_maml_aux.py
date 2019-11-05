#coding: utf-8
import os
import sys
import logging

import clks
import utils

from clks.utils.config import Config
import utils.maml_aux as maml_aux
ModeKeys = maml_aux.ModeKeys

try:
    from .__conf__ import *
    from .config_maml_aux import args
except:
    from __conf__ import *
    from config_maml_aux import args


def train(args):
    config = Config().from_file(args["config"])

    if args["dataset"] == "mwoz20":
        from data.mwoz20.interface_maml_aux import build_meta_input_fn
        input_fn = build_meta_input_fn(
                        args, args["data_path"], training=True)
        main_task_loaders, aux_task_loaders = input_fn(ModeKeys.TRAIN, args, config)
        from models.mwoz20.interface_maml_aux import build_meta_model_fn
        model_fn = build_meta_model_fn(args, config)
        train_spec = model_fn(ModeKeys.TRAIN, 
                main_task_loaders[0], aux_task_loaders)
        eval_spec = model_fn(ModeKeys.EVAL, 
                main_task_loaders[1], aux_task_loaders)
        scf = maml_aux.MamlAuxLearner().build(args, config, 
            os.path.join(ENV_PATH, "maml_aux_trained"), args["run_id"])
        scf.train(train_spec, eval_spec)
    else:
        raise Exception("Unimplemented")


def evaluate(args):
    config = Config().from_file(args["config"])

    if args["dataset"] == "mwoz20":
        from data.mwoz20.interface_maml_aux import build_meta_data_fn
        # (train_loader, dev_loader, test_loader,
        #     test_4d_loader) = prepare_data_loader(
        #         args, args["data_path"], training=False)
        from models.mwoz20.interface_maml_aux import build_meta_model_fn
        model_fn = build_meta_model_fn(args, config)
        # for dev set
        # if args["run_dev_testing"]:
        #     print("Development Set ...")
        #     eval_spec = model_fn(ModeKeys.EVAL, dev_loader)
        #     scf = scaffold.Scaffold().build(args, config,
        #                 os.path.join(ENV_PATH, "trained"), args["run_id"])
        #     scf.evaluate(eval_spec)

        # if args['except_domain'] != "" and args["run_except_4d"]:
        #     print("Test Set on 4 domains...")
        #     eval_spec = model_fn(ModeKeys.EVAL, test_4d_loader)
        #     scf = scaffold.Scaffold().build(args, config,
        #                 os.path.join(ENV_PATH, "trained"), args["run_id"])
        #     scf.evaluate(eval_spec)

        # # for test set
        # print("Test Set ...")
        # eval_spec = model_fn(ModeKeys.TEST, test_loader)
        # scf = scaffold.Scaffold().build(args, config,
        #                                 os.path.join(ENV_PATH, "trained"), args["run_id"])
        # scf.evaluate(eval_spec)
    else:
        raise Exception("Unimplemented")


def test(args):
    raise Exception("Unimplemented")


if __name__ == '__main__':
    os.chdir(MDST_HOME)
    {"train": train, "eval": evaluate, "test": test}[args["mode"]](args)
