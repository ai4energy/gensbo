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
# ======================================================================
# History:
#   
# ======================================================================
# TODO:
#
# ======================================================================

"""

变量模块
  TODO: 修改
"""
from sys import path
path.append('..\\..\\')
from gensbo.tools.util import listattributes

INFINITY = 1.0e+20
"""无穷大"""


class Variable(object):
    """
    优化问题变量类
        * name      (变量名)
        * vartype   (变量类型)
        * lowbound  (变量下界)
        * upbound   (变量上界)
        * value     (当前值)
    """

    def __init__(self,
                 name='x1',
                 vartype='continuous',
                 lowbound=None,
                 upbound=None,
                 set=None,
                 value=1.0,
                 *args,
                 **kwargs):

        self.name = name
        """The name of the variable (str)"""

        self._vartype = vartype
        """The type of the variable (str)"""

        self._lowbound = lowbound
        """The lower boundary of the variable"""

        self._upbound = upbound
        """The upper boundary of the variable"""

        self._set = set
        """The set of values of the variable(非连续离散变量)"""

        self._value = value
        """The (current) value of the variable"""

    def __str__(self):

        # TODO 修改下面的内容，可以给出更易读的变量关键信息
        """
        打印变量的结构信息
        """
        vardescript = 'Variable: %s\n' % (self.name)

        #vardescript = vardescript + '    Name    Type    LowBound    UpBound\n' + '    ' + str(
        #    self.name) + '    %5s    %14.2e    %14.2e    \n' % (
        #        self.vartype, self.lowbound, self.upbound)

        return vardescript

    def __repr__(self):
        return self.__str__

    @property
    def vartype(self):
        """
        变量的类型
            * 'continuous'  (continuous variable)
            * 'binary'      (binary 0/1 variable)
            * 'discrete'     (integer variable)
            * 'discrete-disconti'     (integer variable in a set)
        """
        return self._vartype

    @vartype.setter
    def vartype(self, value):

        # TODO 修改下面的内容，可以在value没输入非常正确完整的情况下可以设置变量

        if not (value in ['binary', 'integer', 'continuous']):
            raise ValueError('unknown variable type')
        self._vartype = value

    @property
    def lowbound(self):
        return self._lowbound

    @lowbound.setter
    def lowbound(self, value):
        self._lowbound = value

    @property
    def upbound(self):
        return self._upbound

    @upbound.setter
    def upbound(self, value):
        self._upbound = value

    @property
    def set(self):
        return self._set

    @upbound.setter
    def set(self, value):
        self._set = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def listattributes(self):
        listattributes(self)


if __name__ == "__main__":
    print("Testing ... ")
    var = Variable('x', vartype='continuous', lowbound=-15.0, upbound=15.0)
    var2 = Variable("bdc", vartype='binary', lowbound=0, upbound=1.0)
    #print(var)
    var.listattributes()

    #print(var2)
    var2.listattributes()