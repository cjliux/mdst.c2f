#coding: utf-8
"""
code massively borrowed from https://github.com/dragen1860/MAML-Pytorch
"""
import os
import sys

import copy
import time
import torch 
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import logging
from clks.utils.log import logger
import clks.utils.scaffold
ModeKeys = clks.utils.scaffold.ModeKeys


# class MyParameter(nn.Parameter):

#     def __init__(self, tensor):
#         self.weight = tensor
#         self._my_grad_fn = tensor._grad_fn

#     def grad_fn(self):
#         return self._my_grad_fn


def hack_model(model, nparams):
    for name, param in nparams.items():
        nlist = name.split('.')
        mod = model
        for n in nlist[:-1]:
            mod = getattr(mod, n)
        # second derivative can be omitted.
        setattr(mod, nlist[-1], nn.Parameter(param))


class MetaLearner(object):

    def __init__(self):
        super().__init__()
        # self.opt = torch.Adam(self.net.parameters(), lr=meta_step_size)
    
    def build(self, args, config, save_path, learner_name):
        self.args, self.config = args, config
        
        self.save_path = save_path
        self.learner_name = learner_name
        self.learner_path = os.path.join(self.save_path, self.learner_name)

        if not os.path.exists(self.learner_path):
            os.makedirs(self.learner_path)

        self.logger = logger
        logger.setLevel(logging.INFO)
        return self

    # def meta_update(self, model, dev_loader, ls, opt):
    #     print('\n Meta update \n')
    #     data = dev_loader.__iter__().next()
    #     # We use a dummy forward / backward pass to get the correct grads into self.net
    #     loss, disp_vals = model.compute_loss(data)
    #     # Unpack the list of grad dicts
    #     gradients = {k: sum(d[k] for d in ls) for k in ls[0].keys()}
    #     # Register a hook on each parameter in the net that replaces the current dummy grad
    #     # with our grads accumulated across the meta-batch
    #     hooks = []
    #     for (k,v) in self.net.named_parameters():
    #         def get_closure():
    #             key = k
    #             def replace_grad(grad):
    #                 return gradients[key]
    #             return replace_grad
    #         hooks.append(v.register_hook(get_closure()))
    #     # Compute grads for current step, replace with summed gradients as defined by hook
    #     opt.zero_grad()
    #     loss.backward()
    #     # Update the net parameters with the accumulated gradient according to optimizer
    #     opt.step()
    #     # Remove the hooks before next training phase
    #     for h in hooks:
    #         h.remove()

    # def train_inner_loop(self, model, opt, train_loader, dev_loader, eval_callback):
    #     # tr_pre_acc = self.callback(self, train_loader)
    #     # val_pre_acc = self.callback(self, dev_loader)

    #     origin = copy.deepcopy(model.state_dict())
    #     for it, date in enumerate(train_loader):
    #         print('inner step', it)
            
    #         loss, disp_vals = model.compute_loss(data)
    #         grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            
    #         hack_model(model, fast_weights)
    #     ##### Test net after training, should be better than random ####
    #     tr_post_acc = eval_callback(self, train_loader)
    #     val_post_acc = eval_callback(self, dev_loader) 

    #     # print('Train Inner step Acc', tr_pre_acc, tr_post_acc)
    #     # print('Val Inner step Acc', val_pre_acc, val_post_acc)

    #     # hack_model(model, origin)
    #     data = dev_loader.__iter__().next()
    #     loss, disp_vals = model.compute_loss(data)
    #     loss = loss / self.meta_batch_size
    #     hack_model(model, origin)
    #     grads = torch.autograd.grad(loss, model.parameters())
    #     meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
    #     metrics = (tr_pre_acc, tr_post_acc, val_pre_acc, val_post_acc)
    #     return metrics, meta_grads

    def evaluate(self, model, train_loader, dev_loader):
        model.eval()


    def train(self, train_spec, eval_spec):
        tasks = train_spec["tasks"]
        model = train_spec["model"]
        optimizer = train_spec["optimizer"]
        meta_optimizer = train_spec["meta_optimizer"]

        self.num_updates = train_spec["num_updates"]
        self.inner_step_size = train_spec["inner_step_size"]
        self.num_inner_updates = train_spec["num_inner_updates"]

        self.callback = eval_spec["callback"]

        opt_init_state = copy.deepcopy(optimizer.state_dict())
        meta_opt_init_state = copy.deepcopy(meta_optimizer.state_dict())

        prev_min_loss, early_stop_count = 1 << 30, train_spec["max_patience"]

        train_time = 0
        for epo in range(self.num_meta_epoch):
            sw = time.time()

            sup_loss, sup_cnt = 0.0, 0

            it_task_batches = {d: (iter(trld), iter(teld)) 
                        for d, (trld, teld) in tasks.items() }
            max_task_steps = min(len(l[0]) for d, l in tasks.items())

            init_state = copy.deepcopy(model.state_dict())

            for ts in range(self.num_task_steps):
                task_batches = {d: (next(trit), next(teit)) 
                    for d, (trit, teit) in it_task_batches.items()}
                
                # for each task
                loss_tasks = []
                for d, (btr, bte) in task_batches:
                    model.load_state_dict(init_state)
                    optimizer.zero_grad()

                    # init_state = copy.deepcopy(model.state_dict())

                    for tmp_grad in range(self.maml_step):
                        loss, _ = model.compute_loss(btr)
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm(model.parameters(),
                            train_spec["grad_clip_value"])
                        optimizer.step()

                    # resample
                    # loss for the meta-update
                    loss, _ = model.compute_loss(bte)
                    loss_tasks.append(loss)

                model.load_state_dict(init_state)
                meta_optimizer.zero_grad()
                loss_meta = torch.stack(loss_tasks).sum() / len(task_batches)
                loss_meta.backward()
                
                grad = torch.nn.utils.clip_grad_norm(
                    model.parameters(), train_spec["meta_grad_clip_value"])
                meta_optimizer.step()

                init_state = copy.deepcopy(model.state_dict())

                sup_loss += loss_meta.item()
                sup_cnt += 1


            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            train_time = time.time() - sw
            self.logger.info("Train time: {}".format(train_time))
            self.logger.info("avg training loss in epoch {} sup: {:.4f}".format(epo, epoch_sup_loss))

            # evaluation
            

            optimizer.load_state_dict(opt_init_state)
            meta_optimizer.load_state_dict(meta_opt_init_state)
            
        return self.model

    def test_inner_loop(self, train_loader, dev_loader):
        mtr_acc, mval_acc = 0.0, 0.0

        for _ in range(self.num_updates):
            # Make a test net with same parameters as our current net
            origin = copy.deepcopy(self.model.state_dict())
            test_opt = torch.optim.SGD(
                self.model.parameters(), lr=self.inner_step_size)

            # Train on the train examples, using the same number of updates as in training
            for i in range(self.num_inner_updates):
                data = self.train_loader.__iter__().next()
                loss, disp_vals = self.model.compute_loss(data)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
            # Evaluate the trained model on train and val examples
            tacc = self.callback(self, train_loader)
            vacc = self.callback(self, dev_loader)
            # mtr_loss += tloss
            mtr_acc += tacc
            # mval_loss += vloss
            mval_acc += vacc

        # mtr_loss = mtr_loss / self.num_updates
        mtr_acc = mtr_acc / self.num_updates
        # mval_loss = mval_loss / self.num_updates
        mval_acc = mval_acc / self.num_updates
        
        self.model.load_state_dict(origin)

    def test(self, test_spec):
        tasks = test_spec["tasks"]
        self.model = test_spec["model"]
        self.callback = test_spec["callback"]
        self.num_updates = test_spec["num_updates"] if "num_updates" in test_spec else 10
        self.num_inner_updates = test_spec["num_inner_updates"]
        self.inner_step_size = test_spec["inner_step_size"]
        
        task_metrics = []
        for d, (train_loader, dev_loader) in tasks.items():
            task_metrics.append(self.test_inner_loop(self.train_loader, self.dev_loader))
        
        # assumed single testing task
        mtr_acc, mval_acc = task_metrics[0] 
        return mtr_acc, mval_acc
