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

"""
utilities
"""


def listattributes(self):
    print("\n")
    print('Attributes List of: ' + repr(self.__dict__['name']) + ' - ' +
          self.__class__.__name__ + ' Instance\n')
    self_keys = sorted(self.__dict__)
    for key in self_keys:
        if key != 'name':
            print(str(key) + ' : ' + repr(self.__dict__[key]))
        # end
    # end
    print('\n')


def subclasses(cls, just_leaf=False):
    sc = cls.__subclasses__()

    ssc = [g for s in sc for g in subclasses(s, just_leaf)]

    return [s for s in sc if not just_leaf or not s.__subclasses__()] + ssc


def subclass_where(cls, **kwargs):
    k, v = next(iter(kwargs.items()))
    for s in subclasses(cls):

        if hasattr(s, k) and getattr(s, k) == v:
            return s

    raise KeyError("No subclasses of {0} with cls.{1} == '{2}'".format(
        cls.__name__, k, v))
