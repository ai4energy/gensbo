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
from sys import path
path.append('..\\..\\')
from gensbo.optimizers.pso import *
# 若添加除pso外的新算法，需要先在此处导入

if __name__ == '__main__':
    s = PSO()
    print("hello",s.name)