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
r"""

基于仿真的优化器，大类

"""
from sys import path
path.append('..\\')
from gensbo.tools.dp import *


__all__ = ["GenSBO"]


class GenSBO(object):
    def __init__(self, problem, simulator, optimizer):
        # 问题
        self.problem = problem
        # 仿真器
        self.simulator = simulator
        # 优化器
        self.optimizer = optimizer
        # 记录当前寻优结果，防止意外报错使得进程结束
        self.result_temp = []

        # 记录寻优过程产生的可行解及其评价函数值
        self.feasible_x_temp = []
        self.feasible_f_temp = [] # 从小到大顺序

    def set_optimizer(self):
        pass

    def set_simulator(self):
        pass

    def run(self):
        # 设置仿真器（引入实例）
        self.optimizer.set_simulator(self.simulator)

        # 记录当前寻优结果，防止意外报错使得进程结束（result功能与此重复）
        self.result_temp.append(self.optimizer.result_temp)

        # 记录寻优过程产生的可行解及其评价函数值
        self.feasible_x_temp.append(self.optimizer.feasible_x_temp)
        self.feasible_f_temp.append(self.optimizer.feasible_f_temp)

        # 调用optimizer获取寻优结果
        self.result = self.optimizer.opt()

    ## 数据处理
    def save_data(self, filename=None, filepath=None):
        _filename = filepath + filename
        _algo_name = self.optimizer.name
        _result = self.result
        _var_dict = self.problem._variables # 字典
        _var_name = list(_var_dict.keys())

        return write_to_file(_algo_name, _result, _filename, var_name = _var_name)

    def visualize(self, filename=None, filepath=None):
        _filename = filepath + filename
        _algo_name = self.optimizer.name
        _result = self.result

        return visualize(_algo_name, _result, _filename)




if __name__ == "__main__":
    # gensbo= GenSBO()
    # gensbo.run()
    print("hello")