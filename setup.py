from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
from distutils.extension import Extension

import re
import glob


ignores = [
    "./**/setup.py", "./setup.py", "./build/**/*.py",
    "**/setup.py", "setup.py", "build/**/*.py"
]
no_compiles = [
    "./**/__init__.py", "./**/__main__.py", "./**/__conf__.py", 
    "./scripts/**/*.py", "./**/scripts/*.py", "./**/scripts/**/*.py",
    "**/__init__.py", "**/__main__.py", "**/__conf__.py", 
    "scripts/**/*.py", "**/scripts/*.py", "**/scripts/**/*.py"
]

all_py_files = glob.glob("./**/*.py", recursive=True) 
py_modules = [file[2:-3] for file in all_py_files
    if any([glob.fnmatch.fnmatch(file, mod_pattern) 
    for mod_pattern in no_compiles])]

setup(
    name = "mdst",
    version = "0.0.1",
    description="",
    author = "cjliux",
    author_email = "cjliux@gmail.com",
    url = "https://cjliux.github.io/",
    license=open("LICENSE", encoding='utf-8').read(),
    ext_modules = cythonize(all_py_files, exclude=ignores+no_compiles),
    py_modules = py_modules
)

