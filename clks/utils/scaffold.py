#coding: utf-8
"""
TODO: unfinished
"""
import os
import sys
import json
import shutil
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from abc import ABCMeta, abstractmethod
import socket
from datetime import datetime
from tqdm import tqdm
import logging

from .log import logger
import clks
import clks.optim
import clks.optim.lr_scheduler
import clks.func.tensor as T


class ModeKeys:
    def __init__(self):
        self.__state__ = {}
    def __getattr__(self, name):
        if name not in self.__state__:
            self.__state__[name] = len(self.__state__)
        return self.__state__[name]
ModeKeys = ModeKeys()


class Scaffold:

    def __init__(self):
        super().__init__()

    def build(self, args, config, save_path, scaffold_name):
        self.args = args
        self.config = config

        self.save_path = save_path
        self.scaffold_name = scaffold_name
        self.scaffold_path = os.path.join(self.save_path, self.scaffold_name)

        if not os.path.exists(self.scaffold_path):
            # build new workspace.
            print("built scaffold at {}".format(self.scaffold_path))
            os.makedirs(self.scaffold_path)

        # # init logger
        log_file = os.path.join(
            self.scaffold_path, "{}.log".format(self.scaffold_name))
        self.logger = logger
        self.logger.setLevel(logging.INFO)
        # formatter = logging.Formatter(
        #     "%(asctime)s - %(levelname)s - %(message)s", 
        #     datefmt='%Y-%m-%d %H:%M:%S')
        # sh = logging.StreamHandler()
        # sh.setLevel(logging.INFO)
        # sh.setFormatter(formatter)
        # self.logger.addHandler(sh)
        return self

    def train(self, train_spec, eval_spec = None):
        train_loader = train_spec["train_loader"]
        self.model = train_spec["model"]
        self.optimizer = train_spec["optimizer"]
        if "lr_scheduler" in train_spec:
            self.lr_scheduler = train_spec["lr_scheduler"]

        do_eval = eval_spec is not None
        if do_eval:
            eval_loader = eval_spec["eval_loader"]
            eval_callback = eval_spec["callback"]

        total_numel = sum(p.numel() for p in self.model.parameters())
        optim_numel = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("{} parameters in total; {} ({:.2f}%) parameters to optimize.".format(
            total_numel, optim_numel, optim_numel/total_numel*100))

        self.ckpt = self.init_ckpt() 
        if self.args["load_ckpt"]:
            self.logger.info("load checkpoint for resumation.")
            
            # load ckpt file if restored
            if self.args["target"] is not None:
                ckpt_file = os.path.join(self.scaffold_path, 
                    "{}.{}.ckpt".format(
                        self.scaffold_name, self.args["target"]))
            else:
                ckpt_file = os.path.join(self.scaffold_path, 
                    "{}.ckpt".format(self.scaffold_name))
            if not os.path.exists(ckpt_file):
                msg = ckpt_file + " doesn't exists."
                self.logger.error(msg)
                raise Exception(msg) 
            self.unpack_ckpt(self.load_by_torch(ckpt_file))
        elif train_spec["do_init"]:
            # init models weights
            self.logger.info("init weights for training the first time.")
            if "init_model" in train_spec:
                self.model.load_state_dict(train_spec["init_model"])
            elif hasattr(self.model, "init_weights"):
                self.model.init_weights()
            
        self.logger.info("start training model {} early stopping.".format(
            "with" if train_spec["max_patience"] > 0 else "without"))
        self.early_stop = False
        # outer loop.
        for epo in range(self.ckpt["curr_epoch"], train_spec["max_epoch"]):
            # inner loop
            if (hasattr(self, "lr_scheduler") and 
                    isinstance(self.lr_scheduler, clks.optim.lr_scheduler.LRScheduler)):
                self.lr_scheduler.step_epoch(self.ckpt["curr_epoch"])
            self.model.train()

            self.model.before_epoch(self.ckpt["curr_epoch"])

            num_batches = len(train_loader)
            # pbar = tqdm(enumerate(train_loader), total=num_batches)
            for it, batch_data in enumerate(train_loader):
                # schedule lr.
                if (hasattr(self, "lr_scheduler") and 
                        isinstance(self.lr_scheduler, clks.optim.lr_scheduler.LRScheduler)):
                    self.lr_scheduler.step_update(self.ckpt["global_count"])

                self.model.train()
                self.optimizer.zero_grad()

                self.model.before_update(self.ckpt["global_count"])

                loss, disp_vals = self.model.compute_loss(batch_data)

                loss.backward()
                if train_spec["clip_grad_norm"]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.trainable_parameters(), 
                        train_spec["grad_clip_value"])
                else:
                    grad_norm = T.total_norm([
                        p.grad.data for p in self.model.trainable_parameters() 
                        if p.grad is not None])
                self.optimizer.step()

                self.ckpt["global_count"] += 1

                train_loss = loss.data.item()

                # periodical display
                if (train_spec["disp_freq"] >= 1
                        and self.ckpt["global_count"] % train_spec["disp_freq"] == 0
                    or train_spec["disp_freq"] > 0 
                        and self.ckpt["global_count"] % max(
                            int(train_spec["disp_freq"] * num_batches), 1) == 0):
                    msg = "[{}][Train] E {} B {}({:.0f}%) U {}; C {:.4f} N {}; ".format(
                        self.scaffold_name, epo, it, it * 100. / num_batches,
                        self.ckpt["global_count"], train_loss, 
                        "na" if grad_norm is None else "{:.4f}".format(grad_norm))
                    for k, v in disp_vals.items():
                        msg += "{} {:.4f} ".format(k, v.item() if isinstance(v, torch.Tensor) else v)
                    self.logger.info(msg)

                if np.isnan(train_loss):
                    self.logger.info("NaN occurred! Stop training.")
                    self.early_stop = True
                    break

                # periodical evaluation
                if do_eval and not train_spec["do_epo_eval"] and (
                        train_spec["eval_freq"] >= 1
                            and self.ckpt["global_count"] % train_spec["eval_freq"] == 0
                        or train_spec["eval_freq"] > 0
                            and self.ckpt["global_count"] % max(
                                int(train_spec["eval_freq"] * num_batches), 1) == 0):
                    with torch.no_grad():
                        curr_eval_score = eval_callback(self, eval_loader)
                    msg = "[{}][Eval] E {} B {} SC {:.4f}".format(
                                    self.scaffold_name, epo, it, curr_eval_score)
                    if self.ckpt["best_eval_score"] is not None:
                        msg += " Best SC {:.4f}".format(self.ckpt["best_eval_score"]) 
                    self.logger.info(msg)

                    # check whether better.
                    if (self.ckpt["best_eval_score"] is None 
                            or not np.isfinite(self.ckpt["best_eval_score"])
                            or curr_eval_score > self.ckpt["best_eval_score"] 
                                and np.isfinite(curr_eval_score)):
                        self.ckpt["best_eval_score"] = curr_eval_score
                        self.ckpt["best_eval_model"] = copy.deepcopy(self.model.state_dict())

                        self.logger.info("saving best eval model.")
                        self.save_by_torch(self.ckpt["best_eval_model"], os.path.join(
                            self.scaffold_path, "{}.best.pth".format(self.scaffold_name)))

                        if train_spec["max_patience"] > 0:
                            self.ckpt["curr_patience"] = 0
                    else:
                        if train_spec["max_patience"] > 0:
                            self.ckpt["curr_patience"] += 1
                    
                    # check early stopping.
                    if (train_spec["max_patience"] > 0
                            and self.ckpt["curr_patience"] >= train_spec["max_patience"]):
                        self.early_stop = True
                        break

                # periodical checkpoint. (for resumation)
                if (train_spec["save_freq"] > 0 
                        and self.ckpt["global_count"] % train_spec["save_freq"] == 0):
                    self.logger.info("auto save new checkpoint.")
                    self.save_by_torch(self.pack_ckpt(), os.path.join(self.scaffold_path, 
                        '{}.ckpt'.format(self.scaffold_name)))
            # epo eval
            if do_eval and train_spec["do_epo_eval"] and (
                    train_spec["eval_freq"] >= 1
                            and (self.ckpt["curr_epoch"] + 1) % train_spec["eval_freq"] == 0
                        or train_spec["eval_freq"] > 0
                            and (self.ckpt["curr_epoch"] + 1) % max(
                                int(train_spec["eval_freq"] * train_spec["max_epoch"]), 1) == 0):
                with torch.no_grad():
                    curr_eval_score = eval_callback(self, eval_loader)
                msg = "[{}][Eval] E {} B {} SC {:.4f}".format(self.scaffold_name, epo, it, curr_eval_score)
                if self.ckpt["best_eval_score"] is not None:
                    msg += " Best SC {:.4f}".format(self.ckpt["best_eval_score"]) 
                self.logger.info(msg)

                if (hasattr(self, "lr_scheduler") 
                    and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                    self.lr_scheduler.step(curr_eval_score)

                # check whether better.
                if (self.ckpt["best_eval_score"] is None or 
                        curr_eval_score > self.ckpt["best_eval_score"]):
                    self.ckpt["best_eval_score"] = curr_eval_score
                    self.ckpt["best_eval_model"] = copy.deepcopy(self.model.state_dict())

                    self.logger.info("saving best eval model.")
                    self.save_by_torch(self.ckpt["best_eval_model"], os.path.join(
                        self.scaffold_path, "{}.best.pth".format(self.scaffold_name)))

                    if train_spec["max_patience"] > 0:
                        self.ckpt["curr_patience"] = 0
                else:
                    if train_spec["max_patience"] > 0:
                        self.ckpt["curr_patience"] += 1

                # check early stopping.
                if (train_spec["max_patience"] > 0
                        and self.ckpt["curr_patience"] >= train_spec["max_patience"]):
                    self.early_stop = True

            if self.early_stop:
                # incomplete epoch doesn't need to be saved, otherwise there'll be bug.
                break

            self.ckpt["curr_epoch"] = epo + 1
            if (hasattr(self, "lr_scheduler") 
                    and isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler)):
                self.lr_scheduler.step()

            # save epoch checkpoint. (for historic performance analysis)
            if train_spec["save_epo_ckpts"]:
                self.logger.info("save checkpoint of epoch {}.".format(epo))
                self.save_by_torch(self.pack_ckpt(), 
                    os.path.join(self.scaffold_path, '{}.epo-{}.ckpt'.format(
                        self.scaffold_name, epo)))

        self.logger.info("save final model") 
        # not the best, in case of eval_freq <= 0.
        if train_spec["save_final_model"]:
            self.save_by_torch(self.model.state_dict(), os.path.join(
                self.scaffold_path, "{}.final.pth".format(self.scaffold_name)))
        self.logger.info("final model saved.")
        return self.model

    def evaluate(self, eval_spec):
        self.model = eval_spec["model"]
        eval_loader = eval_spec["eval_loader"]
        eval_callback = eval_spec["callback"]

        # restore model image.
        if self.args["load_ckpt"]:
            if self.args["target"] is not None:
                file_name = os.path.join(self.scaffold_path, "{}.{}.ckpt".format(
                    self.scaffold_name, self.args["target"]))
            else:
                file_name = os.path.join(self.scaffold_path, "{}.ckpt".format(
                    self.scaffold_name))
            self.logger.info("loading model weights from {}.".format(file_name))
            self.model.load_state_dict(self.load_by_torch(file_name)["best_eval_model"])
        else:
            file_name = os.path.join(self.scaffold_path,  "{}.{}.pth".format(
                self.scaffold_name, self.args["target"]))
            self.logger.info("loading model weights from {}.".format(file_name))
            self.model.load_state_dict(self.load_by_torch(file_name))

        self.logger.info("start evalution.")

        with torch.no_grad():
            eval_callback(self, eval_loader)

        self.logger.info("done.")

    def init_ckpt(self):
        ckpt = {
            "curr_epoch" : 0,
            "global_count" : 0,
            "best_eval_score" : None,
            "best_eval_model" : None,
            "curr_patience" : 0
        }
        return ckpt

    def pack_ckpt(self):
        ckpt = copy.deepcopy(self.ckpt)
        # collect images
        ckpt["model"] = self.model.state_dict()
        ckpt["optimizer"] = self.optimizer.state_dict()
        ckpt["train_stream"] = self.train_stream.state_dict()
        ckpt["eval_stream"] = self.eval_stream.state_dict()
        return ckpt

    def unpack_ckpt(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_stream.load_state_dict(ckpt["train_stream"])
        self.eval_stream.load_state_dict(ckpt["eval_stream"])
        # restore self.ckpt
        for k in self.ckpt.keys():
            self.ckpt[k] = ckpt[k]

    def save_by_torch(self, obj, file_name):
        if os.path.exists(file_name):
            backup_file = file_name+".backup"
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(file_name, backup_file)
        torch.save(obj, file_name)

    def load_by_torch(self, file_name):
        assert os.path.exists(file_name)
        return torch.load(file_name)


if __name__=='__main__':
    scf = Scaffold()
    assert hasattr(scf, "build")

