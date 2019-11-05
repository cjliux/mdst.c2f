#coding: utf-8

# automatically import any Python files in the directory
import os
import importlib
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('clks.optim.' + module)
        