#coding: utf-8
import os
import sys
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_encode(input, vae_trans):
    """
        input: T(...D)
    """
    gauss_params = vae_trans(input)
    gauss_params = torch.clamp(gauss_params, -4, 4)
    mu, logvar = torch.chunk(gauss_params, 2, -1)
    return mu, logvar


class VariationalAutoEncoder(nn.Module):

    def __init__(self, config, name):
        """
        config:
            recog_trans: nn.Linear
            prior_trans: nn.Linear
        """
        super().__init__()
        self.config = config
        self.name = name
        self.default_config()

        self.recog_trans = nn.Linear(**self.config["recog_trans"].todict())
        self.prior_trans = nn.Linear(**self.config["prior_trans"].todict())

    def default_config(self):
        if "freebits" not in self.config:
            self.config["freebits"] = False

    def sample(self, mu, logvar, rand_var=None):
        """
        mu: T(...D)
        logvar: T(...D)
        """
        if rand_var is None:
            rand_var = mu.new(*mu.size()).normal_()
        return mu + rand_var * logvar.exp()

    def recog_encode(self, recog_input):
        """
        input: T(...D)
        """
        return vae_encode(recog_input, self.recog_trans)

    def prior_encode(self, prior_input):
        """
        input: T(...D)
        """
        # gauss_params =  prior_trans(input)
        # gauss_params = torch.clamp(gauss_params, -4, 4)
        # mu, logvar = torch.chunk(gauss_params, 2, 1)
        return vae_encode(prior_input, self.prior_trans)

    def gauss_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        """
        mu/logvar : T(...D)
        """
        loss = ((prior_logvar - recog_logvar) 
            + torch.exp(recog_logvar - prior_logvar) 
            + (prior_mu - recog_mu) ** 2 / torch.exp(prior_logvar) - 1.)
        if self.config["freebits"]:
            mask = loss < self.config["quota"] 
            loss[mask] = loss[mask].detach()
        loss = torch.sum(loss, dim=-1) / 2
        return loss


