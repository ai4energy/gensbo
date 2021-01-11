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
#  contributors  :
#  email         :    mingtao.li@gmail.com
#  url           :    https://www.mingtaoli.cn
#
# ======================================================================
"""
用户函数型的仿真器，用户提供一个problem_function函数，根据varset自变量，
返回目标函数list，约束函数list，

* 以及运行flag # 运行flag是否需要返回？

通过setobjfunc方法设置self._function=objfunc

"""

from gensbo.simulators.simulator import BaseSimulator
from inspect import isfunction

__all__ = ["Trnsys"]


class Trnsys(BaseSimulator):
    name = 'Trnsys'

    def __init__(self, *args, **kwargs):
        self._function = None

    def set_objfunc(self, objfunc):
        """
        调用trnsys仿真器
        :param objfunc:
        :return:
        """
        if isfunction(objfunc):
            self._function = objfunc
        else:
            raise ValueError(
                "Input is not a function for a UserFunction\'s _function\n")
        # end

    def simulate(self, varset):
        """
        评价
        :param varset:
        :return: 平均值，约束值
        """
        print("simulated")
        return self._function(varset)


if __name__ == "__main__":

    myfunc = UserFunction()

    from gensbo.tests.test import problem_function

    # 此处problem_function应该来自类Problem的实例，
    # 即外界产生的problem_function函数应该被包含在Problem的实例当中，再供此处UserFunction的实例调用

    myfunc.set_objfunc(problem_function)

    variables = [0.0] * 3
    variables[0] = 1.0
    variables[1] = 2.0
    variables[2] = 3.0

    objfunctions, constraints = myfunc.simulate(variables)
    print(objfunctions)
    print(constraints)
