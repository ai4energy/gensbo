#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ======================================================================
#
#  gensbo - a GENeral Simulation Based Optimizer
#
#           一个通用的基于仿真的优化器
#
# ======================================================================
#
#  author        :    Mingtao Li
#  date          :    2020.11.26
#  contributors  :    Xiaohai Zhang
#  email         :    mingtao.li@gmail.com
#  github        :    https://github.com/ai4energy/gensbo.git
#
# ======================================================================


import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'),encoding='UTF-8') as f:
    long_description = f.read()

setup(
    name="gensbo",
    version="0.1.0",
    author="Mingtao Li",
    author_email="mingtao.li@gmail.com",
    description="gensbo—a General Simulation Based Optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai4energy/gensbo.git",
    keywords=['optimization', 'simulation', 'pso', 'tool'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
