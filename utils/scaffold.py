#coding: utf-8
import os
import sys
import json
import shutil
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from datetime import datetime

import clks.func.tensor as T

import clks.utils.scaffold
ModeKeys = clks.utils.scaffold.ModeKeys
# Scaffold = clks.utils.scaffold.Scaffold

class Scaffold(clks.utils.scaffold.Scaffold):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

