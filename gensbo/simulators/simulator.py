#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ======================================================================
#
#  gensbo - a GENeral Simulation Based Optimizer
#
#           一个基于仿真的优化器
#
# ======================================================================
#
#  author        :    Mingtao Li
#  date          :    2019.03.26
#  contributors  :    Xiaohai Zhang
#  email         :    mingtao.li@gmail.com
#  url           :    https://www.mingtaoli.cn
#
# ======================================================================

from abc import ABCMeta, abstractmethod


class BaseSimulator(object, metaclass=ABCMeta):
    name = None

    @abstractmethod
    def __init__(self):
        pass

    def simulate(self, varset, if_cal_cons_only=False):
        pass
