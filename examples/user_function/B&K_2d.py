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
#
#  优化问题定义遵循CEC标准：
#       Min f(x), x=[x1,x2,...,xn]  （目标函数）
#       s.t.
#           gi(x) <= 0, i=1,...,q   （不等式约束）
#           hj(x) = 0, j=q+1,...,m  （等式约束）
#       其中，等式约束hj(x)存在允许误差ϵ：
#           |hj(x)|-ϵ <= 0, j=q+1,...,m
#       此处ϵ取1e-5。
#
# ======================================================================
""" 
使用pso方法优化示例

多目标优化：

    此处以2维Binh and Korn function函数为例。

    测试函数信息：
        测试函数名称：Binh and Korn function
        目标函数：
            f1 = 4 * x1 ** 2 + 4 * x2 ** 2
            f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
        约束条件：
            (1) (x1 - 5)**2 + x2**2 - 25 < 0
            (2) -(x1 - 8)**2 - (x2 + 3)**2 + 7.7 < 0
        （注：关于不等式约束，此处"< =" 等同于 "<"，请自行处理）
        自变量个数：2
        自变量范围：x1 - (0,5), x2- (0,3)
        优化目标： min
"""
from sys import path
path.append('..\\..\\')
from gensbo.gensbo import GenSBO
from gensbo.core import Problem
from gensbo.simulators.userfunction import UserFunction
from gensbo.optimizers.pso import PSO
import numpy as np

# ======================================================================
#
#  修改区
#
# ======================================================================
# 
# author        :    Mingtao Li
#  date          :    2019.03.26
#  contributors  :    Xiaohai Zhang
#  email         :    mingtao.li@gmail.com
#  url           :    https://www.mingtaoli.cn
#
# ======================================================================

INFINITY = 1e16  # 无穷大
## 创建优化模型信息

name = "B&K_2d"
problem = Problem(name)

# 变量总数
problem._TotalNumVar = 2
# 约束总数
problem._TotalNumConstraint = 2
# 目标函数总数
problem._NumObjFunc = 2

# 添加变量
###############################################################################################
# 变量类型：                                                                                    #
# 连续变量-"continuous":从[lowbound,upbound]中随机取值(random.uniform(lowbound,upbound))          #
# 离散连续整型变量-"discrete"：从[lowbound,upbound]中随机取整数值(random.randint(lowbound,upbound))  #
# 离散非连续变量-"discrete_disconti"：从传入取值集合（set）中取值                                     #
# 二元整型变量-"binary"：取值0或者1                                                                #
# problem.add_var("xc", "continuous", lowbound=0, upbound=3.0, value=None)                    #
# problem.add_var("xd", "discrete", lowbound=-15.0, upbound=15.0, value=0)                    #
# problem.add_var("xdd", "discrete_disconti", set=[-5, -3, 0, 6, 9, 23], value=6)             #
# problem.add_var("xb", "binary", lowbound=0, upbound=1, value=None)                          #
###############################################################################################
problem.add_var("x1", "continuous", lowbound=0, upbound=5.0, value=2.5)
problem.add_var("x2", "continuous", lowbound=0, upbound=3.0, value=1.5)


# 批量传入变量初始值
# 支持 list 和 np.array 格式
# 每个解向量初值中变量顺序应与添加变量顺序一致（与寻优结果导出的解向量中变量顺序相同）
if_batch_add_var_ini = False
if if_batch_add_var_ini == True:
    # 加载上次寻优导出的可行解
    var_ini = np.load('%s_mo.npy'%name,allow_pickle=True)[0][-1]
    problem.batch_add_var_ini(var_ini)

# 添加目标函数和约束函数
def problem_function(varset,if_cal_cons_only=False):
    """
    添加目标函数和约束函数
    :param varset: 变量集,字典（'var_name':value）
    :param if_cal_cons_only：布尔值，是否只计算约束值而不计算评价函数值，用于产生可行解
    :return: 目标函数值list、约束值list，参考值flag
    """
    objfunc = [0 for _ in range(problem._NumObjFunc)]
    constraint = [[] for _ in range(problem._TotalNumConstraint)]

    # 给变量名赋值（x1 = value)
    globals().update(varset)

    # 添加目标函数
    if if_cal_cons_only == False:
        # 添加目标函数
        objfunc[0] = 4 * x1 ** 2 + 4 * x2 ** 2
        objfunc[1] = (x1 - 5) ** 2 + (x2 - 5) ** 2

    # 添加约束函数
    constraint[0] = ['i', (x1 - 5) ** 2 + x2 ** 2 - 25]
    constraint[1] = ['i', - (x1 - 8) ** 2 - (x2 + 3) ** 2 + 7.7]

    # 参考信息
    flag = 0

    return objfunc, constraint, flag

# 将目标函数和约束函数加入问题实例，存于problem._function
problem.add_objfunc(problem_function)

# ======================================================================
#
#  修改区结束
#
# ======================================================================


# 调用仿真器
# 自定义函数调用优化问题Problem类实例problem的目标函数和约束函数信息
simulator = UserFunction()
simulator.set_objfunc(problem._function)
# f,con = simulator.simulate(varset)

## 优化器选择，创建相应实例
optimizer = PSO(problem)
# 设置仿真器
optimizer.set_simulator(simulator)

# 设置优化器运行参数，未改变的均为默认值
#optimizer.set_options('pso_mode', 'standard_pso')
optimizer.set_options('penalty_type', 'common')

optimizer.set_options('if_mp', False)
optimizer.set_options('mp_core_num', 4)

optimizer.set_options('run_number',3)
optimizer.set_options('x_pareto_size',2000)
optimizer.set_options('if_use_former_x',False)

optimizer.set_options('if_ini_cons', False)
optimizer.set_options('ini_feasible_x_num', 300)

optimizer.set_options('if_get_feasible_x', False)
optimizer.set_options('if_get_feasible_x_only', False)
optimizer.set_options('feasible_x_num', 20)

#print('para',optimizer.options)

if __name__ == "__main__":

    print(problem)

    from time import time
    to = time()

    # 执行主程序
    gensbo = GenSBO(problem, simulator, optimizer)

    # 记录当前寻优结果，防止意外报错使得进程结束
    result_temp = gensbo.result_temp

    # 记录当前寻优可行解结果，防止意外报错使得进程结束
    feasible_f_temp = gensbo.feasible_f_temp
    feasible_x_temp = gensbo.feasible_x_temp

    gensbo.run()

    # 获取寻优结果
    result = gensbo.result

    # 保存数据
    gensbo.save_data(filename=problem.name, filepath='.\\')

    # 结果可视化，若需保存图片则需输入文件名及保存文件路径
    gensbo.visualize(filename=problem.name, filepath='.\\')
    # 求解

    print('time',time()-to)

