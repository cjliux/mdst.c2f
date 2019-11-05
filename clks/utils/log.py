#coding: utf-8
import os
import logging
import logging.config

default_format = "%(asctime)s - %(levelname)s - %(message)s"
default_datefmt = '%Y-%m-%d %H:%M:%S'

logger = logging.getLogger('clks') 
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", 
    datefmt='%Y-%m-%d %H:%M:%S')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.addHandler(sh)
# logger.setLevel(logging.INFO)

