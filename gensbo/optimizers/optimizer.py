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


class BaseOptimizer(object, metaclass=ABCMeta):
    """
    优化器基类

    注意：本类是抽象类，请创建其子类。
    """
    name = None

    def __init__(self):
        """
        创建一个优化器对象
        """
        if self.__class__ is BaseOptimizer:
            raise NotImplementedError("can't instantiate abstract base class")

    def _run(self):
        raise NotImplementedError
