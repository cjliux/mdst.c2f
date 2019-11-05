import os
import sys

import json
import yaml
import torch
from abc import ABCMeta, abstractmethod


class Config():

    def __init__(self, conf = None):
        self._state = conf

    def __iter__(self):
        return iter(self._state)

    def __len__(self):
        return len(self._state)

    def __setitem__(self, path, value, seperator="."):
        if seperator in path:
            path = path.split(seperator)
            conf = self._state
            for p in path[:-1]:
                if p in conf:
                    conf = conf[p]
                else:
                    conf[p] = {}
                    conf = conf[p]
            conf[path[-1]] = value
        else:
            self._state[path] = value

    def __getitem__(self, path, seperator="."):
        return self.extract(path, seperator)

    def extract(self, path, seperator="."):
        try:
            path = path.split(seperator)
            conf = self._state
            for p in path:
                conf = conf[p]
            if isinstance(conf, dict):
                return Config(conf)
            else:
                return conf
        except KeyError as e:
            raise KeyError('.'.join(path))

    def todict(self):
        return self._state

    def keys(self):
        return self._state.keys()

    def values(self):
        values = [self.extract(key) for key in self._state.keys()]
        return values

    def items(self):
        keys = self._state.keys()
        values = [self.extract(key) for key in keys]
        return zip(keys, values)

    def from_file(self, file_name):
        ConfigHandlerFactory(self).get_instance(file_name).load_config()
        return self
    
    def from_dict(self, conf_dict):
        self._state = conf_dict

    def to_file(self, file_name):
        ConfigHandlerFactory(self).get_instance(file_name).save_config()

    def print_to(self, logger):
        js = json.dumps(self._state, indent=2)
        logger.info(js)



class ConfigHandlerFactory(object):
    
    def __init__(self, config):
        self.config = config

    def get_instance(self, file_name):
        ext = os.path.splitext(file_name)[1]
        if ext == '.yaml':
            return YAMLConfigHandler(self.config, file_name)
        elif ext == 'json': 
            return JSONConfigHandler(self.config, file_name)
        else:
            raise NotImplementedError("Unsupported type `{}`".format(ext))


class ConfigHandler(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, file_name):
        self.config = config
        self.file_name = file_name

    @abstractmethod
    def load_config(self):
        pass

    @abstractmethod
    def save_config(self):
        pass


class YAMLConfigHandler(ConfigHandler):
    
    def load_config(self):
        with open(self.file_name, "r", encoding= "utf8") as fd:
            # update config with data
            conf = yaml.safe_load(fd.read())
            self.config._state = conf

    def save_config(self):
        with open(self.file_name, "w", encoding="utf8") as fd:
            # transform config to data
            yaml.dump(self.config._state, fd)


class JSONConfigHandler(ConfigHandler):

    def load_config(self):
        with open(self.file_name, "r", encoding= "utf8") as fd:
            # update config with data
            conf = json.load(fd)
            self.config._state = conf

    def save_config(self):
        with open(self.file_name, "w", encoding="utf8") as fd:
            # transform config to data
            js = json.dumps(self.config._state, indent=2)
            fd.write(js)
