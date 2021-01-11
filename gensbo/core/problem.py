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
# History:
#
# ======================================================================
# TODO:
#
# ======================================================================
"""
文件说明：
    寻优模型信息“类”：目标函数，自变量（解变量），约束。
    创建寻优模型信息，代码将调用“single_ob_pso”包中的单目标粒子群优化算法进行求解。

输出/调用格式：
    self.variable: 变量信息，数组，variable[i] = [type, range]，
                   type[i]: 第i个变量的类型，range[i]: 第i个变量的取值区间/集合
    self.ob_func(x): 目标函数信息，函数形式，由外界导入
        输入：x-变量数值一维数组
        输出：_ob_func: 目标函数一维数组,_ob_func[i]: 第i个目标函数表达式
    self.constraint(x): 约束函数信息，函数形式，由外界导入
        输入：x-变量数值一维数组
        输出：_con: 约束函数信息数组，_con = [type, expr],
                    type[i]: 第i个约束的类型，expr[i]: 第i个约束函数计算值
"""

# import numpy as np

INFINITY = 1e16  # 无穷大

from gensbo.core.variable import Variable
from gensbo.tools.util import listattributes
from inspect import isfunction


class Problem(object):
    """
    优化问题类
    创建一个优化问题对象，包含有问题的变量、目标函数、约束
    """

    def __init__(self, name, *args, **kwargs):
        """
        优化问题初始化
        参数：
        * name -> 字符串，问题的名称
        #* objfunc -> 函数类型，问题的目标函数
        """

        self.name = name
        """问题的名称"""

        self._TotalNumVar = 0
        """总变量数"""

        self.var_ini_list = None
        """传入变量初值"""

        '''
        self._NumBinVar = 0
        """二元0/1变量数"""

        self._NumIntVar = 0
        """连续整数变量数"""

        self._NumIntVarDisconti = 0
        """非连续整数变量数"""

        self._NumContinusVar = 0
        """连续变量数"""
        '''

        self._NumObjFunc = 0
        """目标函数个数"""

        self._TotalNumConstraint = 0
        """总约束个数"""

        '''
        self._NumEquaConstr = 0
        """等式约束个数"""

        self._NumInequaConstr = 0
        """不等式约束个数"""
        '''
        # self.objfunc = objfunc

        self._variables = {}
        """变量字典(dict)"""

        self._objfuncs = []
        """目标函数表(list)"""

        '''
        self._constraints = []
        """约束表(list)：这是一个list"""
        # TODO 先使用list来记录，以后再根据情况看是否需要改变        
        '''

    def __str__(self):
        """
        打印优化问题的结构信息
        """
        probstr = '优化问题: ' + self.name + '\n'
         #   --目标函数%s\n''' % (self.name, self.objfunc.__name__)
        for var in self._variables.keys():
            lines = str(self._variables[var]).split('\n')
            probstr = probstr + lines[0] + '\n'
        return probstr

    def __repr__(self):
        probstr = '优化问题\n'
        probstr += 'optimization problem '
        return probstr

    def add_var(self, name, *args, **kwargs):
        """给优化问题增加一个变量"""
        if name in self._variables:
            raise Exception('this variable already exists')

        if (len(args) > 0) and isinstance(args[0],
                                          Variable) and (name == args[0].name):
            self._variables[name] = args[0]
        else:
            try:
                self._variables[name] = Variable(name, *args, **kwargs)
            except:
                raise ValueError(
                    "Input is not a Valid for a Variable Object instance\n")
            # end
        # end

        # TODO 添加完变量之后，相应的_TotalNumVar等就要发生变化

    def batch_add_var_ini(self,var_ini_list):
        """
        批量传入变量初始值
        :param var_ini_list: 数组，变量初值集合
        :return:
        """
        if len(var_ini_list) <= 0:
            raise ValueError('传入空数组')
        else:
            import numpy as np
            var_ini_list = np.array(var_ini_list)
            if var_ini_list.shape[1] != self._TotalNumVar:
                raise ValueError('传入数据变量值维度存在缺失')

            else:
                if len(var_ini_list.shape) == 1:
                    self.var_ini_list = np.array([var_ini_list])
                else:
                    self.var_ini_list = var_ini_list

    def get_var_ini(self):
        """
        查看传入的变量初值
        :return:
        """
        return self.var_ini_list

    def get_varset(self):
        """
        查看变量集信息
        """
        return self._variables

    def add_objfunc(self, objfunc):
        """
        添加外界产生的优化问题目标函数和约束函数——problem_function函数
        :param objfunc: problem_function函数
        :return:
        """
        # isfunction: Return true if the object is a user-defined function.
        # 如果objfunc是python格式定义的函数形式则返回True
        if isfunction(objfunc):
            self._function = objfunc
        else:
            raise ValueError(
                "Input is not a function for a UserFunction\'s _function\n")

    def add_constraint(self):
        """
        约束信息包含于addobjfunc中，此处暂时未启用。
        :return:
        """
        pass

    def load_fromdict(self, problemdict):
        """
        此处暂时未启用。
        :param problemdict:
        :return:
        """
        print("load from dict")

    @staticmethod
    def load_from_file(filename, problemdict):
        """
        从文件加载问题，此处暂时未启用。
        """
        print("load from dict")
    
    def check(self):
        """
        自我检查，看有没有矛盾的地方，此处暂时未启用。
        """
        pass

    def listattributes(self):
        listattributes(self)


if (__name__ == '__main__'):
    print("hello", "ello")
    prob = Problem("hello")
    print(prob)
    prob.listattributes()
    prob
