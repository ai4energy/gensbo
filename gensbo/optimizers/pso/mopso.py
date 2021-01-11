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

'''
version-0.1.200724_beta
改变：
    (1)增加 if_use_former_x 是否利用前次寻优得到的pareto前沿（稀疏部分）引导本次寻优，默认True


本文件为多目标粒子群算法寻优代码，代码调用用户定义的优化函数信息后返回执行寻优。

本文件包含以下算法：
    （1）全面学习多目标小波粒子群算法

约束处理方式：
    （1）动态罚函数方法（“common”）

优化函数用户定义文件内容说明：（以定义函数形式）

    输出到仿真器的解形式为字典：
    varset = {'var_name1':value1, ...}

以下为本文件相关参数说明：
    所需第三方库：
        numpy, matplotlib

    Algorithm: 定义的算法类，外部文件调用此类将产生一个具体寻优实例

    实例变量：

        产生实例时需外界进行赋值的变量：（所有实例变量均设有默认值，当外界未给定具体值则使用默认值）
            self.swarm_size:        粒子群大小，即粒子数量
            self.w:                 惯性权重
            self.c:                 加速因子
            self.step_max:          最大寻优代数
            self.x_pareto_size:     外部档案（存放非支配解）的大小
            ... (参见doc/userguide)

    函数：

        compare(self, f1, f2)
        ## 比较两个等长数组f1,f2内元素的大小，若f1内元素均小于等于（“<=”）f2内元素，则返回 True,否则返回 False

        delete_feasible(self)
        ## 可行解档案溢出维护:根据f1值的密度（分段）：删除密度大的区域x

        delete_pareto(self, f_pareto, x_pareto)
        ## 外部档案溢出维护，删除欧氏距离小的非劣解

        fitness(self, x, step, penalty_type, penalty_times, oracle)
        ## 计算每个粒子的目标函数值,并将寻优统一为寻最小值以方便操作

        get_feasible_point(self,num_cons_type)
        ## 利用sopso寻找内点

        get_index_disconti(self, xi)
        ## 将离散非连续变量实际值恢复为映射值（在取值集合中的索引号index）

        get_value_disconti(self, xi)
        ## 将离散非连续变量映射值恢复为实际值，寻优运算解空间中该变量的值为对应取值集合中该变量实际数值的索引号

        initialize(self, x, v)
        ## 初始化粒子位置与速度

        nondomainted(self, f1, f2)
        ## 比对两个解（数组）是否互不支配，当两个数组中的对应元素相比有大有小（相等）时，则互不支配

        penalty_constraint(self, particle_i, step, penalty_type, penalty_times)
        ## 根据约束情况实施惩罚

        run(self, pm_mo=0.2, xi_wm_mo=0.5, g_mo=1000, pe_mo=0.4, pl_mo=0.1,
            penalty_type = 'common', penalty_times = 100, w_type='linear', w_range=[0.4,1.2])
        ## 寻优主程序

        run_repeat(self, run_number=10, pm_mo=0.7, xi_wm_mo=0.5, g_mo=1000, pe_mo = 0.4, pl_mo = 0.1,
            penalty_type='common', penalty_times=100,
            w_type='linear', w_range=[0.2, 0.6])
        ## 进行多次寻优合并结果，以得到更全面的pareto前沿

        sort_pareto(self, f_pareto, x_pareto, sort_element=0)
        ## 按第给定个目标函数值大小对f_pareto、x_pareto进行从小到大排序

        transform_f(self, f_pareto)
        ## 转换格式：f[i][j]:第i个解的第[j]个目标函数值 ==> _f[i][j]:第i个目标函数数列中对应第j个解对应的值

        update_x_v_moclpsowm(self, x, v, step, pm_mo=0.7, xi_wm_mo=0.5, g_mo=1000, pe_mo = 0.4, pl_mo = 0.1)
        ## 使用全面学习小波粒子群算法（moclpsowm）

        update_pareto(self, f_pareto, x_pareto)
        ## 更新外部档案，删除当前档案中的劣解

        visualize_pareto(self, _f_pareto, fig_type='scatter')
        ## 可视化寻优结果pareto前沿，仅限目标函数个数为2或3个时使用,使用前请先用函数“transform_f”转化f_pareto数组格式

        x_binary(self, _x, _v)          ## 处理二元型变量（0-1变量）更新

        x_continuous(self, _x, _v)      ## 处理连续型变量更新

        x_discrete(self, _x, _v)         ## 处理离散型变量更新

        x_update(self, _x, _v, _j)      ## 根据变量类型进行更新位置,并进行位置值越界处理（令其等于边界值）

'''

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
import sys

class Algorithm:

    ##初始化参数
    def __init__(self, problem, simulator, swarm_size=30, w_mo=0.4, c_mo=2,
                 step_max=200, x_pareto_size = 100, if_mp=False, mp_core_num=2,
                 if_ini_cons=False, ini_feasible_x_num=1, acc_cons_vio=1e-5,
                 ini_step=30, if_get_feasible_x=False,
                 if_get_feasible_x_only=False, feasible_x_num=100,
                 ini_swarm_size=50,ini_step_max=1000,ini_precision = 1e-3):

        # 获取寻优模型信息
        self.problem = problem
        
        # 获取寻优模型目标函数和约束函数信息
        self.simulator = simulator

        # 获取变量信息
        _variable = self.problem._variables
        self._variable_number = self.problem._TotalNumVar  # 变量个数，整数

        # 获取变量类型和范围（取值上下限），区分离散非连续变量
        self._var_name = []  # 获取变量名称，转换成list
        self._variable_type = []
        self._x_max = []
        self._x_min = []
        self._x_discrete_disconti = []  # 存放离散非连续变量取值集合，数组
        self.ini_value = []  # 获取输入的初值，将其赋予一个粒子，可以加速收敛（初值有参考价值的话）
        # 获取批量传入的变量初值(np.array格式)
        self.var_ini_list = self.problem.var_ini_list

        count = -1  # 记录非连续离散变量的编号
        for i in problem._variables:
            count += 1
            self._var_name.append(i)
            self._variable_type.append(problem._variables[i]._vartype)
            # 处理非连续离散变量
            if (problem._variables[i]._vartype == 'discrete_disconti'):
                self._x_min.append(0)
                self._x_max.append(len(problem._variables[i]._set) - 1)
                self._x_discrete_disconti.append([count, problem._variables[i]._set])
                self.ini_value.append(problem._variables[i]._value)
            else:
                self.ini_value.append(problem._variables[i]._value)
                self._x_min.append(problem._variables[i]._lowbound)
                self._x_max.append(problem._variables[i]._upbound)
        self._variable_number_discrete_disconti = len(self._x_discrete_disconti)
        self.ini_value_true_index = []  # 记录设定初始值的变量序号

        # 增加速度限制
        self._v_max = []
        for i in range(self._variable_number):
            self._v_max.append((self._x_max[i] - self._x_min[i]) / 2)

        # 判断是否存在约束
        if (self.problem._TotalNumConstraint > 0):
            self._constraint_judge = True
        else:
            self._constraint_judge = False

        ## 初始化输入实例变量
        self.swarm_size = swarm_size          # 粒子个数
        self.w = w_mo                         # 惯性权重
        self.c = c_mo                         # 加速度学习因子
        self.step_max = step_max              # 最大寻优步数
        self.x_pareto_size = x_pareto_size    # 外部档案（存放非支配解）大小

        # 记录当前寻优结果，防止意外报错使得进程结束
        self.result_temp = [[],[]]

        ## 记录寻优过程获得的可行解，防止程序意外中断丢失结果
        self.feasible_f_temp = []
        self.feasible_x_temp = []

        # 并行计算：建议当目标函数的计算比较耗时的时候才启用，
        # 否则不同cpu核间的通讯消耗足以抵消并行计算的优势
        # 是否使用并行计算（多进程），默认不使用
        self.if_mp = if_mp
        # 并行计算时使用的cpu核数，默认为2
        self.mp_core_num = mp_core_num

        # 是否初始化内点（可行域）
        self.if_ini_cons = if_ini_cons
        # 内点个数
        self.ini_feasible_x_num = ini_feasible_x_num
        if self.ini_feasible_x_num > self.swarm_size:
            self.ini_feasible_x_num = self.swarm_size

        # 许可误差精度(accuracy of constraint violation):用于等式约束
        self.acc_cons_vio = acc_cons_vio

        # 初始化内点的最大独立寻解次数
        self.ini_step= ini_step

        # 初始化内点中sopso的参数设置
        self.ini_swarm_size = ini_swarm_size
        self.ini_step_max = ini_step_max
        self.ini_precision = ini_precision

        # 是否收集寻优过程产生的可行解及其适应度函数值
        self.if_get_feasible_x = if_get_feasible_x

        # 记录可行解个数上限（整数）
        self.feasible_x_num = feasible_x_num

        # 记录寻优过程中获取的可行解及其适应度值
        self.feasible_x = []
        self.feasible_f = []

        # 当可行解个数满足要求时是否停止寻优：只寻找非重复的可行解
        self.if_get_feasible_x_only = if_get_feasible_x_only
        if self.if_get_feasible_x_only == True:
            self.if_get_feasible_x = True

    ##初始化粒子位置与速度
    def initialize(self):
        '''
        初始化粒子位置与速度
        :return: x,v （位置，速度，数组（浮点数，二维））
        '''

        _x = []
        _v = []

        # 离散非连续变量在所有变量中的索引值
        _var_dis_index_in_self = [self._x_discrete_disconti[index][0]
                                  for index in range(len(self._x_discrete_disconti))]

        # 是否批量导入变量初值
        if self.var_ini_list is None:
            index_begin = 0
        else:
            _var_ini_list = self.var_ini_list.tolist()
            num_var_ini = len(_var_ini_list)
            if num_var_ini <= self.swarm_size:
                _x.extend(_var_ini_list)
                index_begin = num_var_ini
            else:
                index_list = random.sample(range(0,num_var_ini),self.swarm_size)
                _x = [_var_ini_list[i] for i in index_list]
                index_begin = self.swarm_size
            print('T_T 已批量传入变量初值 T_T','\n')

            # 将离散非连续变量的值映射为索引值
            if self._variable_number_discrete_disconti != 0:
                for i in range(len(_x)):
                    _x[i] = self.get_index_disconti(_x[i])

        for i in range(index_begin, self.swarm_size):
            _xi = []

            # 如果外界赋予变量初值，则将外界输入的变量初值赋予第一个粒子
            if (i == index_begin):
                for j in range(self._variable_number):
                    if (self.ini_value[j] != None):
                        # 若出现离散非连续变量（set），则将该初始设定值映射为索引值
                        if self._variable_type[j] == 'discrete_disconti':
                            # 获取j变量在self._x_discrete_disconti中的索引
                            _var_dis_index = _var_dis_index_in_self.index(j)
                            _var_value_set = self._x_discrete_disconti[_var_dis_index][1]

                            # 将初始值转换为索引值
                            _xi.append(_var_value_set.index(self.ini_value[j]))
                        else:
                            _xi.append(self.ini_value[j])

                        self.ini_value_true_index.append(j)

                    else:
                        if (self._variable_type[j] == 'binary'):
                            _xi.append(random.randint(0, 1))
                        elif (self._variable_type[j] == 'discrete') or (self._variable_type[j] == 'discrete_disconti'):
                            _xi.append(random.randint(self._x_min[j], self._x_max[j]))
                        else:
                            _xi.append(random.uniform(self._x_min[j], self._x_max[j]))

            else:
                for j in range(self._variable_number):
                    if (self._variable_type[j] == 'binary'):
                        _xi.append(random.randint(0, 1))
                    elif (self._variable_type[j] == 'discrete') or (self._variable_type[j] == 'discrete_disconti'):
                        _xi.append(random.randint(self._x_min[j], self._x_max[j]))
                    else:
                        _xi.append(random.uniform(self._x_min[j], self._x_max[j]))

            _x.append(_xi)

        # 初始化速度
        for i in range(self.swarm_size):
            _vi = []
            for j in range(self._variable_number):
                _vi.append(random.uniform(-self._v_max[j] / 2, self._v_max[j] / 2))

            _v.append(_vi)

        return _x, _v

    ## 将离散非连续变量映射值恢复为实际值，寻优运算解空间中该变量的值为对应取值集合中该变量实际数值的索引号
    def get_value_disconti(self, xi, _if_ini_0=False):
        '''
        将离散非连续变量映射值恢复为实际值，寻优运算解空间中该变量的值为对应取值集合中该变量实际数值的索引号
        :param xi: 寻优空间解向量，数组
        :param _if_ini_0: 是否为第一个粒子的第一次运算，用于防止外界设定的变量初值被更改
        :return: xi: 恢复离散非连续变量实际值的解向量，用于求解罚函数和目标函数
        '''
        _xi = copy.deepcopy(xi)

        if _if_ini_0 == True:

            # 获取属性为'discrete_disconti'的变量序号
            _dis_var_index = [self._x_discrete_disconti[i][0] for i in range(len(self._x_discrete_disconti))]
            # 删除外界赋予初值的离散非连续变量序号，避免设定的初值被更改
            # 即找到未设定初值的离散非连续变量
            _dis_var_index_new = [i for i in _dis_var_index if i not in self.ini_value_true_index]

            if len(_dis_var_index_new) > 0:
                for _num_var in _dis_var_index_new:
                    _var_value_set = self._x_discrete_disconti[_dis_var_index.index(_num_var)][1]
                    _xi[_num_var] = _var_value_set[int(xi[_num_var])]

        else:
            for i in range(len(self._x_discrete_disconti)):
                # self._x_discrete_disconti[i][0]: 该变量在所有变量中的序号
                # self._x_discrete_disconti[i][1]：该变量的取值集合
                _num_var = self._x_discrete_disconti[i][0]
                _var_value_set = self._x_discrete_disconti[i][1]

                _xi[_num_var] = _var_value_set[int(xi[_num_var])]

        return _xi

    ## 将离散非连续变量实际值恢复为映射值（在取值集合中的索引号index）
    # 在初始化内点得到的可行解处使用
    def get_index_disconti(self, xi, _if_ini_0=False):
        '''
        将离散非连续变量映射值恢复为实际值，寻优运算解空间中该变量的值为对应取值集合中该变量实际数值的索引号
        :param xi: 离散非连续变量实际值的解向量，用于求解罚函数和目标函数，一维数组
        :param _if_ini_0: 是否为第一个粒子的第一次运算，用于防止外界设定的变量初值被更改
        :return: xi: 恢复寻优空间解向量，用于寻优（索引编号[0,1,...,len(dis..)-1]）
        '''
        _xi = copy.deepcopy(xi)

        for i in range(len(self._x_discrete_disconti)):
            # self._x_discrete_disconti[i][0]: 该变量在所有变量中的序号
            # self._x_discrete_disconti[i][1]：该变量的取值集合
            _num_var = self._x_discrete_disconti[i][0]
            _var_value_set = self._x_discrete_disconti[i][1]

            _var_index_calcu = _var_value_set.index(_xi[_num_var])

            _xi[_num_var] = _var_index_calcu

        return _xi

    ## 根据约束情况实施惩罚
    def penalty_constraint(self, particle_i, step,
                           penalty_type='common', penalty_times=100):
        '''
        根据约束情况实施惩罚,暂时只使用普通罚函数
        :param xi:   解向量（第i个粒子的位置信息），数组（浮点数）
        :param step: 当前寻优代数
        :param _f_value: 目标函数值，数组（浮点数）,oracle罚函数使用
        :param penalty_type: 选择罚函数方法，普通（动态）罚函数“common”，oracle罚函数“oracle”
        :param penalty_times: penalty_type = 'common'时生效，惩罚倍数，
                             使违反约束的解受到惩罚后的函数值一定大于全局最优适应函数值
        :return: _penalty_value: 惩罚项大小，浮点数
                 _constraint_violate_index: 违反的约束序号集合，数组（整数）

        '''
        global f_origin, constraint

        _acc = self.acc_cons_vio  # 许可误差精度(accuracy of constraint violation)
        _res = 0     # res函数值初值
        _constraint_violate_index = []  # 用于记录违反的约束序号集合，数组（整数）
        _f_value = f_origin[particle_i] # 数组

        # 获取约束信息
        _constraint = constraint[particle_i]
        _constraint_number = self.problem._TotalNumConstraint  # 约束个数，整数
        _constraint_type = []
        _constraint_value = []
        for i in range(_constraint_number):
            _constraint_type.append(_constraint[i][0])  # 约束类型，'i'(<),'e'(=)
            _constraint_value.append(_constraint[i][1])  # 约束函数计算值，浮点数

        # 计算惩罚函数项大小（_penalty_value）
        _penalty_value = 0

        # 计算res函数值
        for j in range(_constraint_number):
            _value = _constraint_value[j]
            if (_constraint_type[j] == 'less') and (_value > 0):
                _res += _value ** 2
                _constraint_violate_index.append(j)
            elif (_constraint_type[j] == 'equality') and (abs(_value) > _acc):
                _res += _value ** 2
                _constraint_violate_index.append(j)

            _res = np.sqrt(_res)

        # 计算罚函数项值（暂时只使用普通罚函数）
        #if (penalty_type == 'common'):
        _penalty_value = _res ** 2 * (step + 1) * penalty_times

        return _penalty_value, _constraint_violate_index

    ## 计算每个粒子的目标函数值——调用外部用户定义文件,并将寻优统一为寻最小值以方便操作。
    def fitness(self, x, step, penalty_type,penalty_times,
                oracle = 1e9,if_cal_cons_only=False):
        '''
        计算每个粒子的目标函数值——调用外部用户定义文件,并将寻优统一为寻最小值以方便操作
        若存在约束条件，则根据实际情况实施惩罚
        :param x:     粒子位置信息，数组（浮点数，二维）
        :param step:  当前寻优代数
        :param penalty_type: 选择罚函数方法，普通（动态）罚函数“common”，oracle罚函数“oracle”
        :param penalty_times: penalty_type = 'common'时生效，惩罚倍数，
                             使违反约束的解受到惩罚后的函数值一定大于全局最优适应函数值
        :param oracle: penalty_type = 'oracle'时生效，Ω初始值，该值必须大于全局最优适应函数值（此处不使用）
        :param if_cal_cons_only：布尔值，是否只计算约束值而不计算评价函数值，用于产生可行解
        :return: _f: （经惩罚函数处理过的）粒子的适应函数值，数组（浮点数）
                 _constraint_violate_particle: 记录当前寻优代数下违反约束的粒子序号，数组（整数）

        注意：此处返回的是经处理后的函数值，实际上该值即为惩罚函数值（oracle罚函数法）
        @@事实上，oracle罚函数法通过寻找惩罚函数值最小的点来实现对目标函数的寻优。
        '''
        global f_origin, constraint

        _f = []
        _constraint_violate_particle = []  # 记录违反约束的粒子序号

        # 是否使用并行计算
        if (self.if_mp == True):
            # 若存在离散非连续变量，则先获取变量实际值
            _x = copy.deepcopy(x)
            varset_list = []
            for i in range(self.swarm_size):
                if (self._variable_number_discrete_disconti != 0):
                    # 初始化赋值：保证外界传入的初始变量值得以保留
                    if i == 0 and step == 0:
                        _x[i] = self.get_value_disconti(_x[i], _if_ini_0=True)
                    else:
                        _x[i] = self.get_value_disconti(_x[i])

                varset_list.append((dict(zip(self._var_name, _x[i])),
                                    if_cal_cons_only))

            # 并行计算目标函数值与约束值
            with mp.Pool(processes=self.mp_core_num) as pl:
                r = pl.starmap(self.simulator.simulate,varset_list)

            # 处理数据
            for i in range(self.swarm_size):
                _f_value = r[i][0]   # 1D
                f_origin[i] = _f_value    # 2D
                constraint[i] = r[i][1]     # 2D

                if (self._constraint_judge == True):
                    _penalty_value, _constraint_violate_index = self.penalty_constraint(
                        i, step, penalty_type, penalty_times)
                    if (penalty_type == 'common'):
                        for j in range(len(_f_value)):
                            _f_value[j] += _penalty_value

                    # 判断是否违反约束
                    if (len(_constraint_violate_index) != 0):
                        _constraint_violate_particle.append(i)

                _f.append(_f_value)

        else:
            for i in range(self.swarm_size):
                # 若存在离散非连续变量，则先获取变量实际值
                _xi = copy.deepcopy(x[i])

                if (self._variable_number_discrete_disconti != 0):
                    # 初始化赋值：保证外界传入的初始变量值得以保留
                    if i == 0 and step == 0:
                        _xi = self.get_value_disconti(_xi, _if_ini_0=True)
                    else:
                        _xi = self.get_value_disconti(_xi)

                ## 计算目标函数值

                # 将变量名与对应当前值合并为一个新字典
                varset = dict(zip(self._var_name, _xi))

                # 计算目标函数值
                _f_value, constraint[i] = self.simulator.simulate(varset,
                                                                  if_cal_cons_only=if_cal_cons_only)[0:2]
                # _f_value: 目标函数值，数组（多目标有多个元素）
                f_origin[i] = copy.deepcopy(_f_value)

                if (self._constraint_judge == True):
                    _penalty_value, _constraint_violate_index = self.penalty_constraint(
                        i, step, penalty_type, penalty_times)
                    if (penalty_type == 'common'):
                        for j in range(len(_f_value)):
                            _f_value[j] += _penalty_value

                    # 判断是否违反约束
                    if (len(_constraint_violate_index) != 0):
                        _constraint_violate_particle.append(i)

                _f.append(_f_value)

        return _f, _constraint_violate_particle

    ## 不同类型变量位置与速度更新方法 ##

    # 连续型变量-continuous
    def x_continuous(self, _x, _v):
        '''
        连续型变量位置信息更新
        :param _x: 当前位置，数值
        :param _v: 更新后的速度，数值
        :return: _x （更新后的位置，数值）
        '''

        _x = _x + _v

        return _x

    # 二元型变量-binary
    def x_binary(self, _x, _v):
        '''
        二元型变量位置信息更新
        :param _x: 当前位置，数值
        :param _v: 更新后的速度，数值
        :return: _x （更新后的位置，数值）
        '''

        _ro = random.uniform(0, 1)  # 预定阈值
        _sig = 1 / (1 + np.exp(- _v))  # logistic函数(sigmoid曲线（S型曲线）)

        if (_sig > _ro):
            _x = 1
        else:
            _x = 0

        return _x

    # 离散型变量-discrete，采用小数点后四舍五入取整
    def x_discrete(self, _x, _v):
        '''
        离散型变量位置信息更新
        :param _x: 当前位置，数值
        :param _v: 更新后的速度，数值
        :return: _x （更新后的位置，数值）
        '''

        _x = _x + _v

        _x = round(_x)

        return _x

    # 根据变量类型进行更新位置,并进行位置值越界处理（令其等于边界值）
    def x_update(self, _x, _v, _j):
        '''
        根据变量类型进行更新位置,并进行位置值越界处理
        :param _x: 当前位置，数值
        :param _v: 更新后的速度，数值
        :param _j: 第_j个自变量，整数
        :return: _x （更新后的位置，数值）
        '''
        # 速度限制
        if (_v < - self._v_max[_j]):
            _v = - self._v_max[_j]
        elif (_v > self._v_max[_j]):
            _v = self._v_max[_j]

        # 根据变量类型进行更新位置
        if (self._variable_type[_j] == 'continuous'):
            _x = self.x_continuous(_x, _v)
        elif (self._variable_type[_j] == 'binary'):
            _x = self.x_binary(_x, _v)
        elif (self._variable_type[_j] == 'discrete') or (self._variable_type[_j] == 'discrete_disconti'):
            _x = self.x_discrete(_x, _v)
        else:
            print('粒子位置更新时变量类型错误（variable_type_error in updating the position），'
                  '请检查变量类型定义(continuous,binary,discrete)')

        # 处理越界的位置值，此处只按边界值处理
        if (_x < self._x_min[_j]):
            _x = self._x_min[_j]
        elif (_x > self._x_max[_j]):
            _x = self._x_max[_j]

        return _x

    ## 粒子位置与速度更新机制 ##

    ## 使用全面学习多目标小波粒子群算法（moclpsowm）
    def update_x_v_moclpsowm(self, x, v, x_pbest, x_gbest, step,
                             pm_mo=0.7, xi_wm_mo=0.5, g_mo=1000, pe_mo = 0.4, pl_mo = 0.1):
        '''
        使用全面学习多目标小波粒子群算法（moclpsowm）更新粒子的位置
        :param x:     粒子位置信息，数组（浮点数，二维）
        :param v:     粒子速度信息，数组（浮点数，二维）
        :param x_pbest:粒子自身历史最优位置
        :param x_gbest:粒子对应全局最优位置
        :param step:  当前寻优代数
        :param pm_mo:    执行小波变异的概率阈值
        :param xi_wm_mo: 形状参数
        :param g_mo:     a的上限值，常取1000或10000
        :param pe_mo:    精英概率
        :param pl_mo:    学习概率
        :return: x,v （位置，速度，数组（浮点数，二维））
        '''

        for i in range(self.swarm_size):
            _r1 = random.uniform(0, 1)
            _r2 = random.uniform(0, 1)  # 产生（0，1）间的随机数
            _r3 = random.uniform(0, 1)  # produce random number between (0,1)

            # 执行小波变异
            if (_r1 < pm_mo):
                _a = np.exp(- np.log(g_mo) * (1 - step / self.step_max) ** xi_wm_mo) + np.log(g_mo)
                _phi = random.uniform(-2.5 * _a, 2.5 * _a)
                _sigma = 1 / np.sqrt(_a) * np.exp(-(_phi / _a) ** 2 / 2) * np.cos(5 * (_phi / _a))

                if _sigma > 0:
                    for j in range(self._variable_number):
                        x[i][j] = x_gbest[i][j] + _sigma * (self._x_max[j] - x_gbest[i][j])
                else:
                    for j in range(self._variable_number):
                        x[i][j] = x_gbest[i][j] + _sigma * (x_gbest[i][j] - self._x_min[j])

            # 更新速度与位置
            for j in range(self._variable_number):

                if (_r2 > pe_mo):
                    v[i][j] = self.w * v[i][j] + self.c * _r3 * (x_gbest[i][j] - x[i][j])
                elif (_r2 > pl_mo):
                    v[i][j] = self.w * v[i][j] + self.c * _r3 * (x_pbest[i][j] - x[i][j])
                else:
                    v[i][j] = self.w * v[i][j] + self.c * _r3 * (
                            x_pbest[(random.randint(0, self.swarm_size - 1))][j] - x[i][j])

                x[i][j] = self.x_update(x[i][j], v[i][j], j)

        return x, v

    ## 比较两个等长一维数组f1,f2内元素的大小，若f1内元素均小于等于（“<=”）f2内元素，则返回 True,否则返回 False
    def compare(self, f1, f2):
        '''
        比较两个等长一维数组f1,f2内元素的大小，若f1内元素均小于等于（“<=”）f2内元素，则返回 True,否则返回 False
        :param f1: 第一个数组（一维（浮点数））
        :param f2: 第二个数组（一维（浮点数））
        :return: True:  f1中元素均“小于或等于”f2中对应的元素，即f1支配f2
                 False: f1中存在元素“大于”f2中对应的元素
        '''
        _record = 0
        for i in range(len(f1)):
            if (f1[i] <= f2[i]):
                _record += 1
        if (_record == len(f1)):
            return True
        else:
            return False

    ## 比对两个解（数组）是否互不支配，对求最小值问题，当两个数组中的对应元素相比有大有小时，则互不支配
    def nondomainted(self, f1, f2):
        '''
        比对两个解是否互不支配，只有当其中一个数组中的一个元素小于另一个数组中对应元素，而其他元素都是大，则互不支配
        :param f1: 第一个数组（一维（浮点数））
        :param f2: 第二个数组（一维（浮点数））
        :return: True:  两个解互不支配
                 False: 两个解不满足互不支配
        '''

        _count_less = 0
        _count_greater = 0
        for i in range(len(f1)):
            if (f1[i] < f2[i]):
                _count_less += 1
            elif (f1[i] > f2[i]):
                _count_greater += 1
        if (_count_less != 0) and (_count_greater != 0):
            return True
        else:
            return False

    ## 更新外部档案，删除当前档案中的劣解
    def update_pareto(self, f_pareto, x_pareto):
        '''
        更新外部档案，删除当前档案中的劣解
        :param f_pareto:  当前外部档案的目标函数值集合
        :param x_pareto:  当前外部档案的非支配解集合
        :return: f_pareto:  更新后的外部档案的目标函数值集合
                 x_pareto:  更新后的外部档案的非支配解集合
        '''

        _index_pop = []
        # 记录集合中劣解的索引位置，_index_pop[i]:该值表明外部档案中第i个解为当前的劣解，需要踢出外部档案

        # 删除被支配的劣解
        _index_pop = []
        for i in range(len(f_pareto)):
            for j in range(len(f_pareto)):
                if (i != j):
                    if (self.compare(f_pareto[i], f_pareto[j]) == True):
                        _index_pop.append(j)

        _index_pop = np.unique(_index_pop)
        # np.unique()：删去数组中重复元素，并将元素由小到大排列

        _record = 0
        for i in _index_pop:
            f_pareto.pop(i - _record)
            x_pareto.pop(i - _record)
            _record += 1

        return f_pareto, x_pareto

    ## 外部档案溢出维护，删除欧氏距离小的非劣解
    def delete_pareto(self, f_pareto, x_pareto):
        '''
        更新外部档案，删除当前档案中的劣解
        :param f_pareto:  当前外部档案的目标函数值集合
        :param x_pareto:  当前外部档案的非支配解集合
        :return: f_pareto:  更新后的外部档案的目标函数值集合
                 x_pareto:  更新后的外部档案的非支配解集合
        '''
        _over_number = len(f_pareto) - self.x_pareto_size   # 获取溢出个数

        _r = []        # 存放两个连续非劣解间的欧氏距离

        # 按第一个目标函数值大小对f_pareto、x_pareto进行从小到大排序
        f_pareto, x_pareto = self.sort_pareto(f_pareto,x_pareto)

        for i in range(len(f_pareto) - 1):
            _rd = 0
            for j in range(len(f_pareto[0])):
                _rd += (f_pareto[i][j] - f_pareto[i + 1][j]) ** 2
            _r.append(np.sqrt(_rd))

        for i in range(_over_number):
            _index = _r.index(min(_r))
            f_pareto.pop(_index)
            x_pareto.pop(_index)
            _r.pop(_index)

            # 改变因该粒子删除造成的（前一粒子）距离变化
            _rd = 0
            if (_index != 0):
                for j in range(len(f_pareto[0])):
                    _rd += (f_pareto[_index - 1][j] - f_pareto[_index][j]) ** 2
                _r[_index - 1] = np.sqrt(_rd)


        return f_pareto, x_pareto

    ## 可行解档案溢出维护:根据f1值的密度（分段）：删除密度大的区域x
    def delete_feasible(self):
        f1 = [self.feasible_f[i][0]
              for i in range(len(self.feasible_f))]

        num_over = len(self.feasible_x) - self.feasible_x_num
        df = f1[-1] - f1[0]

        if df < self.acc_cons_vio:
            pop_index = random.sample(range(1, len(self.feasible_x)), num_over)
            pop_index.sort()
            count = 0
            for i in pop_index:
                self.feasible_x.pop(i - count)
                self.feasible_f.pop(i - count)
                count += 1
        else:
            if self.feasible_x_num > 50:
                bins_num = 10
            else:
                bins_num = 4

            while num_over > 0:
                hist, bin_edges = np.histogram(f1, bins=bins_num)
                hist_max = max(hist)
                hist_max_index = hist.tolist().index(hist_max)
                index_begin = sum(hist[:hist_max_index])
                index_end = index_begin + hist_max

                # 不删除左、右边界点
                if index_begin == 0:
                    index_begin += 1
                if index_end == len(f1):
                    index_end -= 1

                bins_aver = int(self.feasible_x_num / bins_num)
                dnum = min(num_over, hist_max - bins_aver, index_end - index_begin)
                pop_index = random.sample(range(index_begin, index_end), dnum)

                pop_index.sort()
                count = 0
                for i in pop_index:
                    self.feasible_x.pop(i - count)
                    self.feasible_f.pop(i - count)
                    count += 1

                num_over = len(self.feasible_x) - self.feasible_x_num
                f1 = [self.feasible_f[i][0]
                      for i in range(len(self.feasible_f))]

    ## 利用sopso寻找内点
    def get_feasible_point(self,num_cons_type):
        '''

        :param num_cons_type: 整型，约束类型数（1 或者 2）
        :return:
        '''

        # 使用sopso寻优内点
        from sys import path
        path.append('..\\..\\..\\')
        from gensbo.gensbo import GenSBO
        from gensbo.core import Problem
        from gensbo.simulators.userfunction import UserFunction
        from gensbo.optimizers.pso import PSO

        # problem_ini = copy.deepcopy(self.problem)
        problem_ini = Problem('ini_feasible')
        problem_ini._variables = self.problem._variables
        problem_ini._TotalNumVar = self.problem._TotalNumVar
        problem_ini._NumObjFunc = 1  # num_cons_type
        problem_ini._TotalNumConstraint = self.problem._TotalNumConstraint

        def ini_function(varset, if_cal_cons_only=False):
            cons = self.simulator.simulate(varset, if_cal_cons_only=True)[1]
            if num_cons_type == 1:
                f_cons = sum([cons[i][1] for i in range(len(cons))])
                _f_ini = [f_cons]
            elif num_cons_type == 2:
                cons_value_i = []
                cons_value_e = []
                for i in range(len(cons)):
                    if cons[i][0] == 'e':
                        cons_value_e.append(-self.acc_cons_vio / (cons[i][1]))
                    elif cons[i][0] == 'i':
                        cons_value_i.append(cons[i][1])
                f_cons_1 = sum(cons_value_i)
                f_cons_2 = sum(cons_value_e)  # 归一化？
                _f_ini = [f_cons_1 + f_cons_2]
            else:
                raise ValueError('约束类型超出等式约束“e”和不等式约束“i”')

            return _f_ini, cons

        problem_ini.add_objfunc(ini_function)

        simulator_ini = UserFunction()
        simulator_ini.set_objfunc(problem_ini._function)
        # f,con = simulator.simulate(varset)

        ## 优化器选择，创建相应实例
        optimizer_ini = PSO(problem_ini)
        # 设置仿真器
        optimizer_ini.set_simulator(simulator_ini)

        optimizer_ini.set_options('pso_mode', 'ispsowm')  # standard_pso
        optimizer_ini.set_options('penalty_type', 'oracle')
        optimizer_ini.set_options('swarm_size', self.ini_swarm_size)
        optimizer_ini.set_options('step_max', self.ini_step_max)
        optimizer_ini.set_options('precision', self.ini_precision)
        optimizer_ini.set_options('if_get_feasible_x', True)
        optimizer_ini.set_options('if_get_feasible_x_only', True)
        optimizer_ini.set_options('feasible_x_num', self.ini_feasible_x_num)

        # 执行主程序
        gensbo_ini = GenSBO(problem_ini, simulator_ini, optimizer_ini)
        gensbo_ini.run()

        return gensbo_ini.result[-1]

    ## 启动寻优
    def run(self, if_ini_cons = False, pm_mo=0.7, xi_wm_mo=0.5, g_mo=1000, pe_mo = 0.4, pl_mo = 0.1,
            penalty_type='common', penalty_times=100,
            w_type='linear', w_range=[0.2, 0.6]):
        '''
        启动寻优
        :param pm_mo:    执行小波变异的概率阈值
        :param xi_wm_mo: 形状参数
        :param g_mo:     小波函数中a的上限值，常取1000或10000
        :param pe_mo:    精英概率
        :param pl_mo:    学习概率
        :param penalty_type: 选择罚函数方法，普通（动态）罚函数“common”，oracle罚函数“oracle”
        :param penalty_times: penalty_type = 'common'时生效，惩罚倍数，
                             使违反约束的解受到惩罚后的函数值一定大于全局最优适应函数值
        :param w_type:  惯性权重形式，'fixed'（常量（0.4）），'linear'（随寻优代数由大（w_range[1]）变小（w_range[0]））
        :param w_range：[w_min, x_max]，惯性权重线性变化的最小值和最大值，w_type = 'linear'时生效
        :return: f_pareto: 寻优结果非支配解集（pareto前沿）的目标函数值集合（数组（浮点数））
                 x_pareto: 寻优结果非支配解向量（数组（浮点数））
        '''
        global f_origin, constraint, vio_cons_par

        x = []              # 存放粒子位置信息，数组（二维，浮点数），x[i][j]: 第i个粒子的第j维变量位置值
        v = []              # 存放粒子速度信息，数组（二维，浮点数），x[i][j]: 第i个粒子的第j维变量速度值
        f = []              # 存放粒子当前寻优代数适应函数值，数组（二维，浮点数），f[i][j]: 第i个粒子的第j个适应函数值
        x_pareto = []       # 存放外部档案（pareto前沿）中粒子位置信息，数组（二维，浮点数）
        f_pareto = []       # 存放外部档案（pareto前沿）中粒子适应函数值信息（可能带有惩罚项），数组（二维，浮点数）
        f_pareto_ori = []   # 存放外部档案（pareto前沿）中粒子适应函数值信息（不带有惩罚项），数组（二维，浮点数）
        f_origin = []       # 存放当前粒子的真实目标函数值，数组（浮点数），f_origin[i]: 第i个粒子的真实适应函数值
        constraint = []     # 存放当前粒子的约束函数信息，数组（浮点数），f_origin[i]: 第i个粒子的约束函数信息数组[,]
        vio_cons_par = []   # 存放当前寻优代数下违反约束的粒子序号，数组（整数），用于判断是否需要重新初始化粒子位置
        oracle = 1e9        # oracle罚函数法参数，此处不使用
        f_origin.extend([] for i in range(self.swarm_size))  # 初始化
        constraint.extend([] for i in range(self.swarm_size))  # 初始化

        ## 获取合适的初始化粒子位置、速度（initialize the position and velocity.）
        x, v = self.initialize()
        # 获取粒子适应函数值和当前违反约束的粒子序号集合

        # 若初始化的位置均违反约束则使用sopso优化得到内点
        if self.if_ini_cons == True:

            # 获取粒子适应函数值和当前违反约束的粒子序号集合
            f, vio_cons_par = self.fitness(x, 0, penalty_type, penalty_times, oracle,
                                            if_cal_cons_only=True)
            # vio_cons_par:[i,_constraint_violate_index]
            # constraint 自动更新
            vio_cons_x_index = [vio_cons_par[i][0] for i in range(len(vio_cons_par))]
            cons_type = [constraint[0][i][0] for i in range(len(constraint[0]))]
            num_cons_type = len(np.unique(cons_type)) # 约束类型数量

            count_ini = 0

            if (self._constraint_judge == True):
                print('--------初始化内点开始--------','\n')

                num_feasible_x = self.swarm_size - len(vio_cons_par)

                # 记录sopso寻优内点的初始参数值
                ini_swarm_size = self.ini_swarm_size
                ini_step_max = self.ini_step_max
                ini_precision = self.ini_precision

                while num_feasible_x < self.ini_feasible_x_num:
                    count_ini += 1

                    # 获取寻优结果可行解 feasible_x
                    _x_ini = self.get_feasible_point(num_cons_type)

                    # 更新sopso寻优内点的运行参数
                    self.ini_swarm_size += 10
                    self.ini_step_max *= 1.1
                    self.ini_precision /= 5

                    fea_x_num = len(_x_ini)

                    if fea_x_num > 0:

                        # 还原离散非连续变量的实际值到寻优空间该变量对应的离散值（索引值）
                        if self._variable_number_discrete_disconti != 0:
                            for i in range(fea_x_num):
                                _x_ini[i] = self.get_index_disconti(_x_ini[i])

                        count = 0
                        for i in range(fea_x_num):
                            # 查重
                            if _x_ini[i] not in x:
                                x[vio_cons_x_index[i-count]] = _x_ini[i]
                                vio_cons_x_index.pop(0)
                                num_feasible_x += 1
                                count += 1
                                if len(vio_cons_x_index) == 0:
                                    break
                            if num_feasible_x >= self.ini_feasible_x_num:
                                break

                    print('初始化次数', count_ini,'num_feasible_x',num_feasible_x)
                    if count_ini >= self.ini_step:
                        print('已超过设定初始化内点次数，先寻优试一试','\n')
                        break
            print('--------初始化内点结束--------','\n')

            # 还原sopso寻优内点的运行参数
            self.ini_swarm_size = ini_swarm_size
            self.ini_step_max = ini_step_max
            self.ini_precision = ini_precision

            f, vio_cons_par = self.fitness(x, 0, penalty_type, penalty_times, oracle)
        else:
            f, vio_cons_par = self.fitness(x, 0, penalty_type, penalty_times, oracle)

        ## 若存在离散非连续变量则得到其实际值
        if self._variable_number_discrete_disconti != 0:
            x_true = [self.get_value_disconti(x[i])
                      for i in range(len(x))]
        else:
            x_true = copy.deepcopy(x)

        ## 获取可行解
        if len(vio_cons_par) == 0:
            f_feasible = copy.deepcopy(f_origin)
            x_feasible = copy.deepcopy(x_true)
        elif 0 < len(vio_cons_par) < len(x):
            vio_cons_x_index = [vio_cons_par[i][0] for i in range(len(vio_cons_par))]
            meet_cons_x_index = [i for i in range(len(x)) if i not in vio_cons_x_index]

            x_feasible = [x_true[i] for i in meet_cons_x_index]
            f_feasible = [f_origin[i] for i in meet_cons_x_index]
        else:
            f_feasible = []
            x_feasible = []

        ## 获取非支配解集合
        f_pareto = copy.deepcopy(f_feasible)
        x_pareto = copy.deepcopy(x_feasible)

        # 更新pareto，剔除非支配解 和 违反约束的解
        f_pareto, x_pareto = self.update_pareto(f_pareto, x_pareto)

        # 记录当前pareto前沿，防止程序意外中断
        self.result_temp[0] = f_pareto
        self.result_temp[1] = x_pareto

        # 记录可行解
        if self.if_get_feasible_x == True:
            self.feasible_x.extend(x_feasible)
            self.feasible_f.extend(f_feasible)

            # 判断是否停止:满足可行解个数要求
            num_over = len(self.feasible_x) - self.feasible_x_num
            if num_over >= 0:
                fx_sort = list(zip(self.feasible_f, self.feasible_x))
                fx_sort.sort()  # 小到大排序
                self.feasible_f, self.feasible_x = zip(*fx_sort)
                self.feasible_f = list(self.feasible_f)
                self.feasible_x = list(self.feasible_x)

                # 可行解档案大小维护：删除超出限定个数
                # 根据f值的密度（分段）：删除密度大的区域x
                if num_over > 0:
                    self.delete_feasible()

                # 记录寻优过程产生的可行解及其评价函数值，防止意外报错使得进程结束
                if len(self.feasible_f_temp) != 0:
                    self.feasible_f_temp.pop(0)
                    self.feasible_x_temp.pop(0)

                self.feasible_f_temp.append(self.feasible_f)
                self.feasible_x_temp.append(self.feasible_x)

                if self.if_get_feasible_x_only == True:

                    print('^_^ 已满足可行解个数要求 ^_^', '\n')

                    return f_pareto, x_pareto, \
                           self.feasible_f, self.feasible_x

        ## 获取初始粒子历史最优和全局最优（get the original f_pbest and f_gbest）
        f_pbest = copy.deepcopy(f)  # 获取粒子初始历史最优适应函数值，数组（浮点数）
        x_pbest = copy.deepcopy(x)  # 获取粒子初始历史最优位置信息，用于更新粒子位置

        ## 获取每个粒子对应的全局最优，从外部档案中随机选择
        x_gbest = []
        for i in range(self.swarm_size):
            x_gbest.append(copy.deepcopy(x_pareto[random.randint(0, len(x_pareto) - 1)]))

        ## 更新粒子位置与速度
        x, v = self.update_x_v_moclpsowm(x, v, x_pbest, x_gbest, 1, pm_mo, xi_wm_mo, g_mo, pe_mo, pl_mo)

        ## 开始寻优
        for step in range(1,self.step_max):

            # 根据外界输入命令，判断是否采用随寻优代数线性变化的惯性权重（w）
            if (w_type == 'linear'):
                self.w = w_range[1] - step / self.step_max * (w_range[1] - w_range[0])

            # 获取更新后的粒子适应函数值
            f, vio_cons_par = self.fitness(x, step, penalty_type, penalty_times)

            ## 若存在离散非连续变量则得到其实际值
            if self._variable_number_discrete_disconti != 0:
                x_true = [self.get_value_disconti(x[i])
                          for i in range(len(x))]
            else:
                x_true = copy.deepcopy(x)

            ## 获取可行解
            if len(vio_cons_par) == 0:
                f_feasible = copy.deepcopy(f_origin)
                x_feasible = copy.deepcopy(x_true)
                meet_cons_x_index = [i for i in range(len(x))]
            elif 0 < len(vio_cons_par) < len(x):
                vio_cons_x_index = [vio_cons_par[i][0] for i in range(len(vio_cons_par))]
                meet_cons_x_index = [i for i in range(len(x)) if i not in vio_cons_x_index]

                x_feasible = [x_true[i] for i in meet_cons_x_index]
                f_feasible = [f_origin[i] for i in meet_cons_x_index]
            else:
                f_feasible = []
                x_feasible = []
                meet_cons_x_index = []

            # 更新粒子的历史最优和外部档案
            for i in range(self.swarm_size):
                # 新的位置目标函数值完全支配旧的pbest
                if (self.compare(f[i], f_pbest[i]) == True):
                    f_pbest[i] = copy.deepcopy(f[i])
                    x_pbest[i] = copy.deepcopy(x[i])
                    # 将更好的结果加入外部档案
                    if len(meet_cons_x_index) != 0:
                        if i in meet_cons_x_index:
                            f_pareto.append(copy.deepcopy(f_origin[i]))
                            x_pareto.append(copy.deepcopy(x_true[i]))

                # 新pbest与旧pbest互不支配，则随机选择一个作为新的pbest
                elif (self.nondomainted(f[i], f_pbest[i]) == True):
                    _r = random.uniform(0, 1)
                    if (_r > 0.5):
                        f_pbest[i] = copy.deepcopy(f[i])
                        x_pbest[i] = copy.deepcopy(x[i])
                    if len(meet_cons_x_index) != 0:
                        if i in meet_cons_x_index:
                            f_pareto.append(copy.deepcopy(f_origin[i]))
                            x_pareto.append(copy.deepcopy(x_true[i]))

            # 更新外部档案
            f_pareto, x_pareto = self.update_pareto(f_pareto, x_pareto)

            # 外部档案溢出维护
            if (len(f_pareto) > self.x_pareto_size):
                f_pareto, x_pareto = self.delete_pareto(f_pareto, x_pareto)

            # 暂存结果，防止程序意外中段而丢失已计算结果
            # 此处多余：根据list性质，f_pareto内元素更新将同步到result_temp
            self.result_temp[0] = f_pareto
            self.result_temp[1] = x_pareto

            # 记录可行解
            if self.if_get_feasible_x == True:
                self.feasible_x.extend(x_feasible)
                self.feasible_f.extend(f_feasible)

                # 判断是否停止:满足可行解个数要求
                num_over = len(self.feasible_x) - self.feasible_x_num
                if num_over >= 0:
                    fx_sort = list(zip(self.feasible_f, self.feasible_x))
                    fx_sort.sort()  # 小到大排序
                    self.feasible_f, self.feasible_x = zip(*fx_sort)
                    self.feasible_f = list(self.feasible_f)
                    self.feasible_x = list(self.feasible_x)

                    # 可行解档案大小维护：删除超出限定个数
                    # 根据f值的密度（分段）：删除密度大的区域x
                    if num_over > 0:
                        self.delete_feasible()

                    # 记录寻优过程产生的可行解及其评价函数值，防止意外报错使得进程结束
                    self.feasible_f_temp.append(self.feasible_f)
                    self.feasible_x_temp.append(self.feasible_x)

                    self.feasible_f_temp.pop(0)
                    self.feasible_x_temp.pop(0)

                    if self.if_get_feasible_x_only == True:
                        print('^_^ 已满足可行解个数要求 ^_^', '\n')

                        return f_pareto, x_pareto, \
                               self.feasible_f, self.feasible_x


            # 为每个粒子选取新的全局最优
            if (len(x_pareto) != 0):
                for i in range(self.swarm_size):
                    x_gbest[i] = copy.deepcopy(x_pareto[random.randint(0, len(x_pareto) - 1)])

            # 更新粒子位置与速度
            x, v = self.update_x_v_moclpsowm(x, v, x_pbest, x_gbest, 1, pm_mo, xi_wm_mo, g_mo, pe_mo, pl_mo)

            # 输出当前寻优进度--进度条形式
            self.progress_bar(step, self.step_max)

        # 排序
        f_pareto, x_pareto = self.sort_pareto(f_pareto, x_pareto)

        # 若寻优达到预设最大代数且尚未收敛，则输出以下信息
        if (step == ((self.step_max - 1))):
            print('已达到最大寻优代数')
            
            # 若存在离散非连续变量，则获取变量实际值
            if (self._variable_number_discrete_disconti != 0):
                for i in range(len(x_pareto)):
                    x_pareto[i] = self.get_value_disconti(x_pareto[i])
                
            print('寻优结果pareto前沿为', f_pareto)
            print('相应的解向量为', x_pareto)

        return f_pareto, x_pareto, self.feasible_f, self.feasible_x

    ## 进行多次寻优合并结果，以得到更全面的pareto前沿
    def run_repeat(self, run_number=30, if_ini_cons = False, pm_mo=0.7, xi_wm_mo=0.5, g_mo=1000, pe_mo = 0.4, pl_mo = 0.1,
            penalty_type='common', penalty_times=100,
            w_type='linear', w_range=[0.2, 0.6], if_use_former_x=True):
        '''
        进行多次寻优合并结果，以得到更全面的pareto前沿
        :param run_number:   独立寻优次数
        :param pm_mo:           执行小波变异的概率阈值
        :param xi_wm_mo:        形状参数
        :param g_mo:            小波函数中a的上限值，常取1000或10000
        :param pe_mo:           精英概率
        :param pl_mo:           学习概率
        :param penalty_type: 选择罚函数方法，普通（动态）罚函数“common”，oracle罚函数“oracle”
        :param penalty_times: penalty_type = 'common'时生效，惩罚倍数，
                             使违反约束的解受到惩罚后的函数值一定大于全局最优适应函数值
        :param w_type:  惯性权重形式，'fixed'（常量（0.4）），'linear'（随寻优代数由大（w_range[1]）变小（w_range[0]））
        :param w_range：[w_min, x_max]，惯性权重线性变化的最小值和最大值，w_type = 'linear'时生效
        :param if_use_former_x:Boolean，是否利用前次寻优得到的pareto前沿（稀疏部分）引导本次寻优，默认True
        :return: f_pareto: 寻优结果非支配解集（pareto前沿）的目标函数值集合（数组（浮点数））
                 x_pareto: 寻优结果非支配解向量（数组（浮点数））
        '''
        result_total = []          # 存放各次运行的结果

        if self.if_get_feasible_x_only == True:
            run_number = 1

        for i in range(run_number):
            print('独立寻优次数',i+1)
            result = self.run(if_ini_cons = if_ini_cons, pm_mo=pm_mo, xi_wm_mo=xi_wm_mo, g_mo=g_mo,
                                         pe_mo=pe_mo, pl_mo=pl_mo,
                                         penalty_type=penalty_type, penalty_times=penalty_times,
                                         w_type=w_type, w_range=w_range)
            f_pareto_i,x_pareto_i = result[:2]
            result_total.append([f_pareto_i,x_pareto_i])

            if if_use_former_x == True:
                bins_num = min(10,int(len(f_pareto_i)/5))
                # 10个以上进行划分
                if bins_num > 1:
                    f1 = [f_pareto_i[j][0] for j in range(len(f_pareto_i))]
                    hist, bin_edges = np.histogram(f1, bins=bins_num)
                    hist_min = min(hist)
                    hist_min_index = hist.tolist().index(hist_min)
                    index_begin = sum(hist[:hist_min_index])
                    index_end = index_begin + hist_min
                    # 获取引导解（稀疏解）
                    self.var_ini_list = np.array(x_pareto_i[index_begin:index_end])

                # 10个以下全部使用
                else:
                    self.var_ini_list = np.array(x_pareto_i)

                #print('self.var_ini_list',len(self.var_ini_list))

        # 汇总非支配解寻优结果
        f_pareto_total = []
        x_pareto_total = []
        for i in range(run_number):
            for j in range(len(result_total[i][0])):
                f_pareto_total.append(result_total[i][0][j])
                x_pareto_total.append(result_total[i][1][j])

        # 更新非支配解集
        f_pareto_total, x_pareto_total = self.update_pareto(f_pareto_total, x_pareto_total)

        # 排序
        f_pareto_total, x_pareto_total = self.sort_pareto(f_pareto_total, x_pareto_total)

        return f_pareto_total, x_pareto_total,\
               self.feasible_f, self.feasible_x

    ## 按第给定个目标函数值大小对f_pareto、x_pareto进行从小到大排序
    def sort_pareto(self, f_pareto, x_pareto, sort_element=0):
        '''
        按第给定个目标函数值大小对f_pareto、x_pareto进行从小到大排序
        :param f_pareto:  当前外部档案的目标函数值集合
        :param x_pareto:  当前外部档案的非支配解集合
        :param sort_element: 需要排序的目标函数序号（从0开始计数）
        :return: f_pareto, x_pareto: 排序后的结果
        '''

        _sort = []  # 方便对f_pareto、x_pareto排序
        for i in range(len(f_pareto)):
            _sort.append([f_pareto[i], x_pareto[i]])

        # 进行排序
        _sort = sorted(_sort, key=lambda x:x[:][0][sort_element])

        for i in range(len(f_pareto)):
            f_pareto[i] = _sort[i][0]
            x_pareto[i] = _sort[i][1]

        return f_pareto, x_pareto

    ## 转换格式：f[i][j]:第i个解（粒子）的第[j]个目标函数值——_f[i][j]:第i个目标函数数列中对应第j个解（粒子）对应的值
    def transform_f(self, f_pareto):
        '''
        转换格式：f[i][j]:第i个解（粒子）的第[j]个目标函数值——_f[i][j]:第i个目标函数数列中对应第j个解（粒子）对应的值
        :param f_pareto:  当前外部档案的目标函数值集合
        :return: _f_pareto:  更新格式后的外部档案的目标函数值集合

        '''
        _f_pareto = []
        _f_pareto.extend([] for i in range(len(f_pareto[0])))

        # 转换格式
        for i in range(len(f_pareto)):
            for j in range(len(_f_pareto)):
                _f_pareto[j].append(f_pareto[i][j])

        return _f_pareto

    ## 可视化寻优结果pareto前沿，仅限目标函数个数为2或3个时使用,使用前请先用函数“transform_f”转化f_pareto数组格式
    def visualize_pareto(self, _f_pareto, fig_type='scatter'):
        '''
        可视化寻优结果pareto前沿，仅限目标函数个数为2或3个时使用,使用前请先用函数“transform_f”转化f_pareto数组格式
        :param _f_pareto: 转换格式后的外部档案的目标函数值集合
        :param fig_type:  画图方式，“scatter”-散点图，“plot”-折线图，三维只支持画散点图
        :return: 二维/三维图
        '''
        # 二维画图
        if (len(_f_pareto) == 2):
            if (fig_type == 'scatter'):
                plt.scatter(_f_pareto[0], _f_pareto[1])
            elif (fig_type == 'plot'):
                plt.plot(_f_pareto[0], _f_pareto[1])

            plt.xlabel("f1(x)")
            plt.ylabel("f2(x)")
            plt.title("Pareto Front")
            plt.show()
        # 三维画图
        elif (len(_f_pareto) == 3):
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(_f_pareto[0], _f_pareto[1], _f_pareto[2])
            plt.show()

    ## 进度条输出
    def progress_bar(self, num, total):
        rate = float(num)/total
        ratenum = int(100*rate)
        r = '\r[{}{}]{}%'.format('*'*ratenum, ' '*(100 - ratenum), ratenum)
        sys.stdout.write(r)
        sys.stdout.flush()

if (__name__ == '__main__'):

    from sys import path
    path.append('..\..\core')
    import model

    m = model.Model(variable_number=2, ob_func_number=2, constraint_number=2)

    # 变量信息
    x = m.variable  # variable为数组
    x[0] = ['continuous', [0, 5]]
    x[1] = ['continuous', [0, 3]]


    # x[2] = ['discrete', [-15, 15]]
    # x[3] = ['discrete_disconti', [-5, -3, 0, 6, 9, 23]]

    # 目标函数信息,通过自定义函数增加进Model类当中
    def ob_func(self, x):
        '''
        :param x: 一维变量数组
        :return: _ob_func: 目标函数一维数组,_ob_func[i]: 第i个目标函数表达式
        '''
        _ob_func = []
        _ob_func.extend([] for i in range(self.ob_func_number))
        #############  修改区  #############

        # 目标函数表达式
        _ob_func[0] = 4 * x[0] ** 2 + 4 * x[1] ** 2  # + x[3]
        _ob_func[1] = (x[0] - 5) ** 2 + (x[1] - 5) ** 2  # + x[2]

        #############  修改区  #############
        return _ob_func


    # 将定义的目标函数加入实例m
    m.ob_func = ob_func


    # 约束信息,通过自定义函数增加进Model类当中
    def constraint(self, x):
        '''
        :param x: 一维变量数组
        :return:_con: 约束函数信息数组，_con = [type, expr], type[i]: 第i个约束的类型，expr[i]: 第i个约束函数计算值
        '''
        constraint_func = []
        constraint_func.extend([] for i in range(self.constraint_number))
        constraint_type = []
        constraint_type.extend([] for i in range(self.constraint_number))
        #############  修改区  #############

        # 约束函数表达式
        constraint_func[0] = (x[0] - 5) ** 2 + x[1] ** 2 - 25
        constraint_func[1] = - (x[0] - 8) ** 2 - (x[1] + 3) ** 2 + 7.7
        # constraint_func[2] = (x[3] + x[2]) + 19

        # 约束形式：'less', 'equality'
        constraint_type[0] = 'less'
        constraint_type[1] = 'less'
        # constraint_type[2] = 'less'

        #############  修改区  #############
        return [constraint_type, constraint_func]


    # 将定义的约束函数加入实例m
    m.constraint = constraint

    problem = m  # 模型信息“类”，由外界导入

    test = Algorithm(problem, swarm_size=30, w_mo=0.4, c_mo=2,
                     step_max=300, x_pareto_size=100)
    result = test.run(pm_mo=0.7, xi_wm_mo=0.5, g_mo=1000, pe_mo=0.4, pl_mo=0.1,
                      penalty_type='common', penalty_times=100,
                      w_type='linear', w_range=[0.2, 0.6])


    f_pareto_sort = result[0]
    x_pareto_sort = result[1]


    # 转换数组格式，可视化结果
    f_pareto_visual = test.transform_f(f_pareto_sort)
    test.visualize_pareto(f_pareto_visual,fig_type='scatter')
    '''
    # 进行多次计算合并结果，以得到更全面的pareto前沿
    result_total = test.run_repeat(run_number=10, pm_mo=0.7, xi_wm_mo=0.5, g_mo=1000, pe_mo=0.4, pl_mo=0.1,
                      penalty_type='common', penalty_times=100,
                      w_type='linear', w_range=[0.2, 0.6])

    f_pareto_total_sort = result_total[0]
    x_pareto_total_sort = result_total[1]

    # 转换数组格式，可视化结果
    f_pareto_total_visual = test.transform_f(f_pareto_total_sort)
    test.visualize_pareto(f_pareto_total_visual,fig_type='scatter')

    '''


    def get_f_pareto_origin(x_pareto, x_pareto_old, f_pareto_ori, x, f_origin):

        # 第一次赋值
        if (len(f_pareto_ori) == 0):
            _x = copy.deepcopy(x)
            _f = copy.deepcopy(f_origin)
            count = 0
            for i in range(len(x_pareto)):
                for j in range(len(_x)):
                    j -= count
                    if (x_pareto[i] == _x[j]):
                        f_pareto_ori.append(f_origin[j])
                        _x.pop(j)
                        _f.pop(j)
                        count += 1

        else:
            _x_p = x_pareto
            _x_p_o = x_pareto_old
            counti = 0
            conutj = 0
            for i in range(len(_x_p)):
                i -= counti
                for j in range(len(_x_p_o)):
                    j -= count
                    if (_x_p[i] == _x_p_o[j]):
                        _x_p.pop(i)
                        _x_p_o.pop(j)