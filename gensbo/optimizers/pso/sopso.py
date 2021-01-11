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
#  date          :    2019.04.18
#  contributors  :    Xiaohai Zhang
#  email         :    mingtao.li@gmail.com
#  url           :    https://www.mingtaoli.cn
#
# ======================================================================

'''
version-0.1.200724_beta
改变：
    (1)增加寻优结果解向量对应的约束函数值输出


本文件为单目标粒子群算法寻优代码，代码调用用户定义的优化函数信息后返回执行寻优。

本文件包含以下算法：
    （1）标准粒子群算法；
    （2）改进小波简化粒子群算法，小波函数为Morlet；

## 算法使用组合建议：
（1）目标函数：较为简单(变量少)-“standard_pso”，较复杂（变量多，极值点多）-“ispsowm”(多样性强，目前局部寻优能力较弱)
（2）约束条件：较为简单-“common”，较复杂-“oracle”

优化函数用户定义文件内容说明：（以定义函数形式）

    输出到仿真器的解形式为字典：
    varset = {'var_name1':value1, ...}

以下为本文件相关参数说明：
    所需第三方库：
        numpy, matplotlib

    Algorithm: 定义的算法类，外部文件调用此类将产生一个具体寻优实例

    实例变量：

        产生实例时需外界进行赋值的变量：（所有实例变量均设有默认值，当外界未给定具体值则使用默认值）
            self.swarm_size:  粒子群大小，即粒子数量
            self.w:           惯性权重
            self.c1:          加速因子中的认知因子
            self.c2:          加速因子中的社会因子
            self.step_max:    最大寻优代数
            self.precision:   收敛精度
            ... (参见doc/userguide)

    函数：

        delete_feasible(self)
        ## 可行解档案溢出维护:根据f1值的密度（分段）：删除密度大的区域x

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

        neighborhood_ring(self, f_gbest, f_pbest)
        ## 使用环型（ring）邻域结构更新各粒子的全局最优

        neighborhood_star(self, f_gbest, f_pbest)
        ## 使用全互连型（star）邻域结构更新各粒子的全局最优

        penalty_constraint(self, particle_i, step, penalty_type, penalty_times, oracle)
        ## 根据约束情况实施惩罚

        run(self, pso_model='standard_pso', neighborhood='star', pm=0.2, xi_wm=0.5, g=1000,
            penalty_type = 'common', penalty_times = 100, oracle = 1e9, w_type='linear', w_range=[0.4,1.2])
        ## 寻优主程序

        update_ipsowm(self, x, v, step, pm=0.2, xi_wm=0.5, g=1000)
        ## 用ipsowm（小波粒子群算法）更新粒子位置与速度

        update_standard_pso(self, x, v)
        ## 用standard_pso（标准粒子群算法）更新粒子位置与速度

        visualize(self, _f_history, fig_type='scatter')
        ## 将寻优过程全局最优函数值的变化过程可视化

        x_binary(self, _x, _v)          ## 处理二元型变量（0-1变量）更新

        x_continuous(self, _x, _v)      ## 处理连续型变量更新

        x_discrete(self, _x, _v)         ## 处理离散型变量更新

        x_update(self, _x, _v, _j)      ## 根据变量类型进行更新位置,并进行位置值越界处理（令其等于边界值）

'''

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import time
import sys
import multiprocessing as mp

class Algorithm:

    ##初始化参数
    def __init__(self, problem, simulator, swarm_size=30, w=0.8, c1=2, c2=2,
                 step_max=500, precision=1e-3, if_mp=False, mp_core_num=2,
                 if_ini_cons=False,ini_feasible_x_num=1,acc_cons_vio=1e-5,
                 ini_step=30,if_get_feasible_x=False,
                 if_get_feasible_x_only=False,feasible_x_num=100,
                 ini_swarm_size=50,ini_step_max=1000,ini_precision = 1e-3):

        # 获取寻优模型变量信息，目标函数个数，约束函数个数
        self.problem = problem

        # 获取寻优模型目标函数和约束函数信息
        self.simulator = simulator

        # 获取变量信息
        _variable = self.problem._variables  # problem._variable类Variable类的一个实例，type为字典（dict）
        self._variable_number = self.problem._TotalNumVar  # 变量个数，整数

        # 获取变量类型和范围（取值上下限），区分离散非连续变量
        self._var_name = []             # 获取变量名称，转换成list
        self._variable_type = []
        self._x_max = []
        self._x_min = []
        self._x_discrete_disconti = []  # 存放离散非连续变量取值集合，数组
        self.ini_value = []             # 获取输入的初值，将其赋予一个粒子，可以加速收敛（初值有参考价值的话）
        # 获取批量传入的变量初值(np.array格式)
        self.var_ini_list = self.problem.var_ini_list

        count = -1  # 记录非连续离散变量的编号
        for i in problem._variables:
            # i:变量名
            count += 1
            self._var_name.append(i)
            self._variable_type.append(problem._variables[i]._vartype)
            # 处理非连续离散变量
            if (problem._variables[i]._vartype == 'discrete_disconti'):
                self._x_min.append(0)
                self._x_max.append(len(problem._variables[i]._set) - 1)
                self._x_discrete_disconti.append([count,problem._variables[i]._set])
                self.ini_value.append(problem._variables[i]._value)
            else:
                self.ini_value.append(problem._variables[i]._value)
                self._x_min.append(problem._variables[i]._lowbound)
                self._x_max.append(problem._variables[i]._upbound)
        self._variable_number_discrete_disconti = len(self._x_discrete_disconti)
        self.ini_value_true_index = []   # 记录设定初始值的变量序号

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
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.step_max = int(step_max)
        self.precision = precision

        # 记录当前寻优结果，防止意外报错使得进程结束
        self.result_temp = []
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

        # 初始化内点的最大运算次数
        self.ini_step = ini_step

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

        for i in range(index_begin,self.swarm_size):
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

    ## 将离散非连续变量映射值恢复为实际值，
    # 寻优运算解空间中该变量的值为对应取值集合中该变量实际数值的索引号
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
                           penalty_type='common', penalty_times=100, oracle=1e9):
        '''
        根据约束情况实施惩罚
        :param particle_i:   第i个粒子
        :param step: 当前寻优代数
        :param penalty_type: 选择罚函数方法，普通（动态）罚函数“common”，oracle罚函数“oracle”
        :param penalty_times: penalty_type = 'common'时生效，惩罚倍数，使违反约束的解受到惩罚后的函数值一定大于全局最优适应函数值
        :param oracle: penalty_type = 'oracle'时生效，Ω初始值，该值必须大于全局最优适应函数值
        :param get_vio_list: 是否获取该解违反的约束序号集合，“yes”获取，“no”不获取
        :return: _penalty_value: 惩罚项大小，浮点数
                 _constraint_violate_index: 违反的约束序号集合，数组（整数）

        '''
        global f_origin, constraint

        _acc = self.acc_cons_vio  # 许可误差精度(accuracy of constraint violation)
        _res = 0     # res函数值初值
        _constraint_violate_index = []  # 用于记录违反的约束序号集合，数组（整数）
        _f_value = f_origin[particle_i]
        _constraint = constraint[particle_i]

        # 获取约束信息
        _constraint_number = self.problem._TotalNumConstraint  # 约束个数，整数
        _constraint_type = []
        _constraint_value = []
        for i in range(_constraint_number):
            _constraint_type.append(_constraint[i][0])    # 约束类型，'i'(<),'e'(=)
            _constraint_value.append(_constraint[i][1])   # 约束函数计算值，浮点数

        # 计算惩罚函数项大小（_penalty_value）
        _penalty_value = 0

        # 计算res函数值
        for j in range(_constraint_number):
            _value = _constraint_value[j]
            if (_constraint_type[j] == 'i') and (_value > 0):
                _res += _value ** 2
                _constraint_violate_index.append(j)
            elif (_constraint_type[j] == 'e') and (abs(_value) > _acc):
                _res += _value ** 2
                _constraint_violate_index.append(j)

            _res = np.sqrt(_res)

        # 计算罚函数项值
        if (penalty_type == 'common'):
            _penalty_value = _res ** 2 * (step + 1) * penalty_times

        elif (penalty_type == 'oracle'):
            if (_f_value <= oracle) and (_res <= _acc):
                _penalty_value = _f_value - oracle

            else:
                if (_f_value <= oracle):
                    _penalty_value = _res

                else:
                    if (_res < ((_f_value - oracle) / 3)):
                        _alpha = ((_f_value - oracle) * (6 * np.sqrt(3) - 2) / (6 * np.sqrt(3)) - _res) / (
                                _f_value - oracle - _res)
                    elif (_res >= ((_f_value - oracle) / 3)) and (_res <= (_f_value - oracle)):
                        _alpha = 1 - 1 / (2 * np.sqrt((_f_value - oracle) / _res))
                    elif (_res > (_f_value - oracle)):
                        _alpha = 0.5 * np.sqrt((_f_value - oracle) / _res)

                    _penalty_value = _alpha * (_f_value - oracle) + (1 - _alpha) * _res

        return _penalty_value, _constraint_violate_index

    ## 计算每个粒子的目标函数值——调用外部用户定义文件,并将寻优统一为寻最小值以方便操作。
    def fitness(self, x, step, penalty_type, penalty_times,
                oracle, if_cal_cons_only=False):
        '''
        计算每个粒子的目标函数值——调用外部用户定义文件,并将寻优统一为寻最小值以方便操作
        若存在约束条件，则根据实际情况实施惩罚
        :param x:     粒子位置信息，数组（浮点数，二维）
        :param step:  当前寻优代数
        :param penalty_type: 选择罚函数方法，普通（动态）罚函数“common”，oracle罚函数“oracle”
        :param penalty_times: penalty_type = 'common'时生效，惩罚倍数，使违反约束的解受到惩罚后的函数值一定大于全局最优适应函数值
        :param oracle: penalty_type = 'oracle'时生效，Ω初始值，该值必须大于全局最优适应函数值
        :param if_cal_cons_only：布尔值，是否只计算约束值而不计算评价函数值，用于产生可行解
        :return: _f: （经惩罚函数处理过的）粒子的适应函数值，数组（浮点数）
                 _constraint_violate_particle: 记录当前寻优代数下违反约束的粒子序号，数组（整数）

        注意：此处返回的是经处理后的函数值，实际上该值即为惩罚函数值（oracle罚函数法）
        @@事实上，oracle罚函数法通过寻找惩罚函数值最小的点来实现对目标函数的寻优。
        '''
        #t0 = time.clock()

        global f_origin, if_calcu_f, constraint

        _f = []
        _constraint_violate_particle = []  # 记录违反约束的粒子序号，及违反的约束编号

        # 是否使用并行计算
        if (self.if_mp == True) and(if_calcu_f == True):
            # 若存在离散非连续变量，则先获取变量实际值
            _x = copy.deepcopy(x)
            varset_list = []
            for i in range(len(x)):
                if (self._variable_number_discrete_disconti != 0):
                    # 初始化赋值：保证外界传入的初始变量值得以保留
                    if i == 0 and step == 0:
                        _x[i] = self.get_value_disconti(_x[i], _if_ini_0=True)
                    else:
                        _x[i] = self.get_value_disconti(_x[i])

                if (if_calcu_f == True):
                    # 将变量名与对应当前值合并为一个新字典
                    varset_list.append((dict(zip(self._var_name,_x[i])),
                                       if_cal_cons_only))

            # 并行计算目标函数值与约束值

            with mp.Pool(processes=self.mp_core_num) as pl:
                r = pl.starmap(self.simulator.simulate,varset_list)
            '''
            pl = mp.Pool(processes=self.mp_core_num)
            r = pl.starmap(self.simulator.simulate, varset_list)
            pl.close()
            pl.join()
            '''

            # 处理数据
            for i in range(len(x)):
                _f_value = r[i][0][0]   # 浮点数
                f_origin[i] = _f_value    # 1D
                constraint[i] = r[i][1]     # 2D

                if (self._constraint_judge == True):
                    _penalty_value, _constraint_violate_index = self.penalty_constraint(i,
                        step, penalty_type, penalty_times, oracle)
                    if (penalty_type == 'common'):
                        _f_value += _penalty_value
                    elif (penalty_type == 'oracle'):
                        _f_value = _penalty_value

                    # 判断是否违反约束
                    if (len(_constraint_violate_index) != 0):
                        _constraint_violate_particle.append([i,_constraint_violate_index])

                _f.append(_f_value)

        else:
            for i in range(len(x)):
                # 若存在离散非连续变量，则先获取变量实际值
                _xi = copy.deepcopy(x[i])

                if (self._variable_number_discrete_disconti != 0):
                    # 初始化赋值：保证外界传入的初始变量值得以保留
                    if i == 0 and step == 0:
                        _xi = self.get_value_disconti(_xi, _if_ini_0=True)
                    else:
                        _xi = self.get_value_disconti(_xi)

                if (if_calcu_f == True):
                    # 将变量名与对应当前值合并为一个新字典
                    varset = dict(zip(self._var_name,_xi))

                    # 计算目标函数值
                    _f_value, constraint[i] = self.simulator.simulate(varset,
                                                                      if_cal_cons_only=if_cal_cons_only)[0:2]
                    # _f_value: 目标函数值，数组（单目标只有一个元素）
                    _f_value = _f_value[0] # 浮点数
                    f_origin[i] = copy.deepcopy(_f_value)

                if (self._constraint_judge == True):
                    _penalty_value, _constraint_violate_index = self.penalty_constraint(i,
                        step, penalty_type, penalty_times, oracle)
                    if (penalty_type == 'common'):
                        _f_value += _penalty_value
                    elif (penalty_type == 'oracle'):
                        _f_value = _penalty_value

                    # 判断是否违反约束
                    if (len(_constraint_violate_index) != 0):
                        _constraint_violate_particle.append([i,_constraint_violate_index])
                else:
                    _f_value = _f_value

                _f.append(_f_value)

        #t1 = time.clock()
        #print('sopso-step',step,'sopso-fitness-time',t1 - t0)

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

    ## 使用标准粒子群算法（standard_pso）
    def update_standard_pso(self, x, v):
        '''
        使用标准粒子群算法（standard_pso）更新粒子的位置与速度
        :return: x,v （位置与速度，数组（浮点数，二维））
        '''

        global x_pbest, x_gbest

        for i in range(self.swarm_size):

            _r1 = random.uniform(0, 1)
            _r2 = random.uniform(0, 1)  # produce random number

            for j in range(self._variable_number):
                v[i][j] = self.w * v[i][j] + self.c1 * _r1 * (x_pbest[i][j] - x[i][j]) + self.c2 * _r2 * (
                            x_gbest[i][j] - x[i][j])

                x[i][j] = self.x_update(x[i][j], v[i][j], j)

        return x, v

    ## 使用改进小波简化粒子群算法（ispsowm）
    def update_ispsowm(self, x, v, step, c=0.1, pm=0.2, xi_wm=0.5, g=1000):
        '''
        使用改进小波简化粒子群算法（ispsowm）更新粒子的位置（简化了速度项）
        :param c:     反向搜索阈值
        :param pm:    执行小波变异的概率阈值
        :param xi_wm: 形状参数
        :param g:     a的上限值，常取1000或10000
        :return: x,v （位置，速度，数组（浮点数，二维））
        '''

        global x_pbest, x_gbest

        for i in range(self.swarm_size):
            _r1 = random.uniform(0, 1)
            _r2 = random.uniform(0, 1)  # 产生（0，1）间的随机数
            _r3 = random.uniform(0, 1)  # produce random number between (0,1)

            # 执行小波变异
            if (_r1 < pm):
                _a = np.exp(- np.log(g) * (1 - step / self.step_max) ** xi_wm) + np.log(g)
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

                # 判断是否反向搜索
                if (_r3 <= c):
                    sign_r3 = -1
                else:
                    sign_r3 = 1

                ##ISPSO
                if (self._variable_type[j] != 'binary'):

                    # 连续、离散型变量
                    if (_r3 <= c):
                        x[i][j] = _r2 * sign_r3 * x[i][j] + (1 - _r2) * self.c1 * _r1 * (x_pbest[i][j] - x[i][j]) + (
                                1 - _r1) * self.c2 * (1 - _r2) * (x_gbest[i][j] - x[i][j])
                    else:
                        x[i][j] = self.w * x[i][j] + (1 - _r2) * self.c1 * _r1 * (x_pbest[i][j] - x[i][j]) + (
                                1 - _r1) * self.c2 * (x_gbest[i][j] - x[i][j])

                    if (self._variable_type[j] == 'discrete') or (self._variable_type[j] == 'discrete_disconti'):
                        x[i][j] = round(x[i][j])

                    # 处理越界的位置值，令其为边界值或一定概率对越界位置在变量允许范围内随机赋值（与反向搜索概率阈值相同）
                    if (x[i][j] < self._x_min[j]):
                        if (_r3 <= c):
                            x[i][j] = random.uniform(self._x_min[j], self._x_max[j])
                        else:
                            x[i][j] = self._x_min[j]
                    elif (x[i][j] > self._x_max[j]):
                        if (_r3 <= c):
                            x[i][j] = random.uniform(self._x_min[j], self._x_max[j])
                        else:
                            x[i][j] = self._x_max[j]

                elif (self._variable_type[j] == 'binary'):
                    # 二元型变量
                    v[i][j] = _r2 * sign_r3 * v[i][j] + (1 - _r2) * self.c1 * _r1 * (x_pbest[i][j] - x[i][j]) + (
                            1 - _r1) * self.c2 * (1 - _r2) * (x_gbest[i][j] - x[i][j])
                    x[i][j] = self.x_update(x[i][j], v[i][j], j)

        return x, v

    ## 不同邻域结构 ##
    # 全互连型-star
    def neighborhood_star(self, f_gbest, f_pbest):
        '''
        使用全互连型（star）邻域结构更新各粒子的全局最优
        :param f_gbest: 上代粒子全局最优适应函数值，浮点数
        :param f_pbest: 各粒子对应的自身历史最优适应函数值，数组（浮点数）
        :return: f_gbest （更新后的全局最优适应函数值，数值）
        '''

        global x_gbest, x_pbest

        if f_gbest > min(f_pbest):
            f_gbest = min(f_pbest)
            index_f_gbest = f_pbest.index(f_gbest)

            for i in range(self.swarm_size):
                x_gbest[i] = copy.deepcopy(x_pbest[index_f_gbest])

        return f_gbest

    # 环型-ring
    def neighborhood_ring(self, f_gbest, f_pbest):
        '''
        使用环型（ring）邻域结构更新各粒子的全局最优
        :param f_gbest: 上代粒子全局最优适应函数值，浮点数
        :param f_pbest: 各粒子对应的自身历史最优适应函数值，数组（浮点数）
        :return: f_gbest （更新后的全局最优适应函数值，数值）
        '''

        global x_gbest, x_pbest

        ##环形拓扑结构（按例子序号，非按粒子实际相邻位置）
        # 第1个粒子
        f_gbest = min([f_pbest[-1], f_pbest[0], f_pbest[1]])
        index_f_gbest = f_pbest.index(f_gbest)
        x_gbest[0] = copy.deepcopy(x_pbest[index_f_gbest])

        # 第2~swarm_size-1个粒子
        for i in range(1, self.swarm_size - 1):
            f_gbest = min([f_pbest[i - 1], f_pbest[i], f_pbest[i + 1]])
            index_f_gbest = f_pbest.index(f_gbest)
            x_gbest[i] = copy.deepcopy(x_pbest[index_f_gbest])

        # 第n个粒子
        f_gbest = min([f_pbest[self.swarm_size - 2], f_pbest[self.swarm_size - 1], f_pbest[0]])
        index_f_gbest = f_pbest.index(f_gbest)
        x_gbest[-1] = copy.deepcopy(x_pbest[index_f_gbest])

        f_gbest = min(f_pbest)

        return f_gbest

    ## 可行解档案溢出维护:根据f值的密度（分段）：删除密度大的区域x
    def delete_feasible(self):

        num_over = len(self.feasible_x) - self.feasible_x_num
        df = self.feasible_f[-1] - self.feasible_f[0]

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
                hist, bin_edges = np.histogram(self.feasible_f, bins=bins_num)
                hist_max = max(hist)
                hist_max_index = hist.tolist().index(hist_max)
                index_begin = sum(hist[:hist_max_index])
                index_end = index_begin + hist_max

                # 不删除左边界点：保留最优解
                if index_begin == 0:
                    index_begin += 1

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

    ## 利用sopso寻找内点
    def get_feasible_point(self, num_cons_type):
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
    def run(self, pso_model='standard_pso', neighborhood='star',
            c=0.1, pm=0.2, xi_wm=0.5, g=1000,
            penalty_type='common', penalty_times=100, oracle=1e9,
            w_type='linear', w_range=[0.4, 1.2]):
        '''
        启动寻优
        :param pso_model:     选择pso算法，‘standard_pso’,'ispsowm'
        :param neighborhood： 选择粒子群邻域结构，'star'（全互连型），'ring'（环型）
        :return: f_result, x_result, constraint_violate_index, f_history
                （数值（浮点数），数组（浮点数），数组（整数），数组（浮点数））
        '''

        global x_pbest, x_gbest, f_origin, if_calcu_f, constraint

        x = []  # 存放粒子位置信息，数组（二维，浮点数），x[i][j]: 第i个粒子的第j维变量位置值
        v = []  # 存放粒子速度信息，数组（二维，浮点数），x[i][j]: 第i个粒子的第j维变量速度值
        f = []  # 存放粒子当前寻优代数的适应函数（目标函数）值，数组（浮点数），f[i]: 第i个粒子的适应函数值
        f_origin = []       # 存放当前粒子的真实目标函数值，数组（浮点数），f_origin[i]: 第i个粒子的真实适应函数值
        constraint = []     # 存放当前粒子的约束函数信息，数组（浮点数），constraint[i]: 第i个粒子的约束函数信息数组[type,value]
        _vio_cons_par = []  # 存放当前寻优代数下违反约束的粒子序号及违反约束，数组，用于判断是否需要重新初始化粒子位置
        self.f_history = []      # 存放全局最优适应函数值的变化历史，数组（浮点数），self.f_history[i]: 第i代全局最优适应函数值
        x_history = []      # 存放全局最优解的变化历史，数组（浮点数），x_history[i]: 第i代全局最优解（向量）
        _constraint_violate_index = []  # 存在约束时，存放违反的约束序号

        f_origin.extend(0 for i in range(self.swarm_size))  # 初始化
        constraint.extend([] for i in range(self.swarm_size))  # 初始化
        if_calcu_f = True  # 是否调用外界目标函数（仿真器）进行评价，减少调用次数(针对使用oracle罚函数方法时）
        update_oracle = False

        ## 获取合适的初始化粒子位置、速度（initialize the position and velocity.）
        x, v = self.initialize()

        # 若初始化的位置均违反约束则使用sopso优化得到内点
        if self.if_ini_cons == True:

            # 获取粒子适应函数值和当前违反约束的粒子序号集合
            f, _vio_cons_par = self.fitness(x, 0, penalty_type, penalty_times, oracle,
                                            if_cal_cons_only=True)
            # _vio_cons_par:[i,_constraint_violate_index]
            # constraint 自动更新
            vio_cons_x_index = [_vio_cons_par[i][0] for i in range(len(_vio_cons_par))]
            cons_type = [constraint[0][i][0] for i in range(len(constraint[0]))]
            num_cons_type = len(np.unique(cons_type))   # 约束类型数量

            count_ini = 0

            if (self._constraint_judge == True):
                print('--------初始化内点开始--------','\n')

                num_feasible_x = self.swarm_size - len(_vio_cons_par)

                while num_feasible_x < self.ini_feasible_x_num:
                    count_ini += 1

                    # 获取寻优结果可行解 feasible_x
                    _x_ini = self.get_feasible_point(num_cons_type)

                    # 更新sopso寻优内点的运行参数
                    self.ini_swarm_size += 10
                    self.ini_step_max *= 1.1
                    self.ini_precision /= 5

                    # 查重
                    fea_x_num = len(_x_ini)

                    if fea_x_num > 0:

                        # 还原离散非连续变量的实际值到寻优空间该变量对应的离散值（索引值）
                        if self._variable_number_discrete_disconti != 0:
                            for i in range(fea_x_num):
                                _x_ini[i] = self.get_index_disconti(_x_ini[i])

                        count = 0
                        for i in range(fea_x_num):
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
            f, _vio_cons_par = self.fitness(x, 0, penalty_type, penalty_times, oracle)
        else:
            f, _vio_cons_par = self.fitness(x, 0, penalty_type, penalty_times, oracle)

        ## 获取初始粒子历史最优和全局最优（get the original f_pbest and f_gbest）
        f_pbest = copy.deepcopy(f)  # 获取粒子初始历史最优适应函数值，数组（浮点数）
        x_pbest = copy.deepcopy(x)  # 获取粒子初始历史最优位置信息，用于更新粒子位置
        f_gbest = min(f_pbest)
        x_gbest = copy.deepcopy(x_pbest) # x_gbest: 数组（二维），x_gbest[i][j]: 第i个粒子的第j维变量的全局最优位置值

        # 选择邻域结构，更新各粒子对应的全局最优（最优函数值和最优解位置）
        if (neighborhood == 'star'):
            f_gbest = self.neighborhood_star(f_gbest, f_pbest)
        elif (neighborhood == 'ring'):
            f_gbest = self.neighborhood_ring(f_gbest, f_pbest)

        # 获取全局最优位置信息
        index_f_gbest = f_pbest.index(f_gbest)
        self.x_gbest_step = copy.deepcopy(x_pbest[index_f_gbest])

        # 获取并记录当前真实全局最优适应函数值（不含惩罚项）和最优位置
        self.f_value = f_origin[index_f_gbest]   # self.f_value:真实当前全局最优适应函数值（不含惩罚项）
        self.f_history.append(self.f_value)
        x_history.append(self.x_gbest_step)

        # 若存在离散非连续变量，则获取变量实际值
        if (self._variable_number_discrete_disconti != 0):
            self.x_gbest_step = self.get_value_disconti(self.x_gbest_step)
            ##=================================================
            # 若当前最佳值包含设定的离散非连续变量初始值，将报错
            ##=================================================

        # 约束存在的情况
        if (self._constraint_judge == True):
            if (penalty_type == 'oracle'):
                index_vio_cons = [_vio_cons_par[i][0] for i in range(len(_vio_cons_par))]
                # 更新oracle值，加惩罚项后的值（P(x)）<0时
                if (f_gbest < 0) and (index_f_gbest not in index_vio_cons):
                    update_oracle = True
                    oracle = self.f_value
                    # 每次成功更新oracle即抛弃之前oracle对应的信息
                    if_calcu_f = False  # 是否调用外界目标函数（仿真器）进行评价，减少调用次数
                    f, _vio_cons_par = self.fitness(x, 0, penalty_type, penalty_times, oracle)

                    f_pbest = copy.deepcopy(f)
                    f_gbest = min(f_pbest)

                    if_calcu_f = True

        # 获取可行解
        if self.if_get_feasible_x == True:

            if len(_vio_cons_par) < len(x):
                vio_cons_x_index = [_vio_cons_par[i][0] for i in range(len(_vio_cons_par))]
                meet_cons_x_index = [i for i in range(len(x)) if i not in vio_cons_x_index]

                self.feasible_x = [copy.deepcopy(x[i]) for i in meet_cons_x_index]
                self.feasible_f = [f_origin[i] for i in meet_cons_x_index]

                # 变换离散非连续变量的索引值（寻优空间）为实际值
                if (self._variable_number_discrete_disconti != 0):
                    for i in range(len(self.feasible_x)):
                        self.feasible_x[i] = self.get_value_disconti(self.feasible_x[i])

                # 判断是否停止:满足可行解个数要求
                if len(self.feasible_x) >= self.feasible_x_num:
                    fx_sort = list(zip(self.feasible_f,self.feasible_x))
                    fx_sort.sort() # 小到大排序
                    self.feasible_f,self.feasible_x = zip(*fx_sort)
                    self.feasible_f = list(self.feasible_f)
                    self.feasible_x = list(self.feasible_x)

                    # 此处不进行可行解档案大小的维护

                    if self.if_get_feasible_x_only == True:
                        self.f_value = self.feasible_f[0]
                        self.x_gbest_step = self.feasible_x[0]
                        _constraint_violate_index = []
                        self.f_history = [self.f_value]

                        _constraint_value_all = [constraint[index_f_gbest][i][1]
                                                 for i in range(len(constraint[index_f_gbest]))]
                        _constraint_info = [_constraint_violate_index, _constraint_value_all]

                        print('^_^ 已满足可行解个数要求 ^_^','\n')

                        return self.f_value, self.x_gbest_step,\
                               _constraint_info, self.f_history,\
                               self.feasible_f, self.feasible_x

        # 记录寻优过程产生的可行解及其评价函数值，防止意外报错使得进程结束
        self.feasible_f_temp.append(self.feasible_f)
        self.feasible_x_temp.append(self.feasible_x)

        ## 更新粒子位置与速度
        if (pso_model == 'standard_pso'):
            x, v = self.update_standard_pso(x, v)
        elif (pso_model == 'ispsowm'):
            x, v = self.update_ispsowm(x, v, 1, c, pm, xi_wm, g)

        ## 开始寻优
        for step in range(1,self.step_max):

            # 根据外界输入命令，判断是否采用随寻优代数线性变化的惯性权重（linear）or混沌惯性权重(chaos)
            if (w_type == 'linear'):
                self.w = w_range[1] - step / self.step_max * (w_range[1] - w_range[0])
            elif (w_type == 'chaos'):  # logistic映射
                self.w = 4 * self.w * (1 - self.w)
                if self.w <= 0 or self.w >= 1:
                    self.w = random.uniform(1e-5,1-1e-5)
                elif self.w in [0.25, 0.5, 0.75]:
                    self.w = random.uniform(1e-5, 1 - 1e-5)


            # 获取更新后的粒子适应函数值
            f, _vio_cons_par = self.fitness(x, step, penalty_type, penalty_times, oracle)

            # 更新粒子的个体历史最优
            for i in range(self.swarm_size):
                if f_pbest[i] > f[i]:
                    f_pbest[i] = copy.deepcopy(f[i])
                    x_pbest[i] = copy.deepcopy(x[i])

            # 选择邻域结构，更新各粒子对应的全局最优（最优函数值和最优解位置）
            if (neighborhood == 'star'):
                f_gbest = self.neighborhood_star(f_gbest, f_pbest)
            elif (neighborhood == 'ring'):
                f_gbest = self.neighborhood_ring(f_gbest, f_pbest)

            # 获取全局最优位置信息
            index_f_gbest = f_pbest.index(f_gbest)
            self.x_gbest_step = copy.deepcopy(x_pbest[index_f_gbest])

            # 若存在离散非连续变量，则获取变量实际值
            if (self._variable_number_discrete_disconti != 0):
                self.x_gbest_step = self.get_value_disconti(self.x_gbest_step)

            # 约束存在的情况
            if (self._constraint_judge == True):
                # 获取当前最优位置目标函数值（不含惩罚项），通过判断全局最优解向量（位置）是否发生变化进行赋值
                if (x_history[-1] != self.x_gbest_step):
                    self.f_value = f_origin[index_f_gbest]
                else:
                    self.f_value = self.f_history[-1]

                if (penalty_type == 'oracle'):
                    index_vio_cons = [_vio_cons_par[i][0] for i in range(len(_vio_cons_par))]
                    # 更新oracle值，加惩罚项后的值（P(x)）<0时
                    if (f_gbest < 0) and (index_f_gbest not in index_vio_cons):
                        update_oracle = True
                        oracle = self.f_value
                        #print('step',step,'oracle',oracle,'f_o_index',f_origin[index_f_gbest],'index',index_f_gbest)
                        # 每次成功更新oracle即抛弃之前oracle对应的信息
                        if_calcu_f = False   # 是否调用外界目标函数（仿真器）进行评价，减少调用次数
                        f, _vio_cons_par = self.fitness(x, step + 1, penalty_type, penalty_times, oracle)

                        f_pbest = copy.deepcopy(f)
                        f_gbest = min(f_pbest)

                        if_calcu_f = True
            # 约束不存在的情况
            elif (self._constraint_judge == False):
                self.f_value = f_gbest

            # 获取可行解
            if self.if_get_feasible_x == True:

                if len(_vio_cons_par) < len(x):

                    vio_cons_x_index = [_vio_cons_par[i][0] for i in range(len(_vio_cons_par))]
                    meet_cons_x_index = [i for i in range(len(x)) if i not in vio_cons_x_index]

                    feasible_x_step = [copy.deepcopy(x[i]) for i in meet_cons_x_index]
                    feasible_f_step = [f_origin[i] for i in meet_cons_x_index]

                    # 变换离散非连续变量的索引值（寻优空间）为实际值
                    if (self._variable_number_discrete_disconti != 0):
                        for i in range(len(feasible_x_step)):
                            feasible_x_step[i] = self.get_value_disconti(feasible_x_step[i])

                    # 判断是否与已记录的可行解重复
                    for i in range(len(feasible_x_step)):
                        if feasible_x_step[i] not in self.feasible_x:
                            self.feasible_x.append(feasible_x_step[i])
                            self.feasible_f.append(feasible_f_step[i])

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

                        if self.if_get_feasible_x_only == True:
                            if index_f_gbest in vio_cons_x_index:
                                _constraint_violate_index = _vio_cons_par[
                                    vio_cons_x_index.index(index_f_gbest)][1]
                            else:
                                _constraint_violate_index = []
                            self.f_history.append(self.f_value)

                            _constraint_value_all = [constraint[index_f_gbest][i][1]
                                                     for i in range(len(constraint[index_f_gbest]))]
                            _constraint_info = [_constraint_violate_index, _constraint_value_all]

                            print('^_^ 已满足可行解个数要求 ^_^','\n')

                            return self.f_value, self.x_gbest_step, \
                                   _constraint_info, self.f_history, \
                                   self.feasible_f, self.feasible_x

            # 记录历史全局最优适应函数值和最优位置
            self.f_history.append(self.f_value)
            x_history.append(self.x_gbest_step)
            self.result_temp.append([self.f_value, self.x_gbest_step])

            # 记录寻优过程产生的可行解及其评价函数值，防止意外报错使得进程结束
            self.feasible_f_temp.append(self.feasible_f)
            self.feasible_x_temp.append(self.feasible_x)

            self.feasible_f_temp.pop(0)
            self.feasible_x_temp.pop(0)

            # 终止迭代条件：种群粒子的自身历史最好目标函数值（集合）的标准差小于收敛精度
            if (np.std(f_pbest) < self.precision):
                print('达到收敛精度要求,寻优代数为',(step + 2),'\n')
                break

            # 更新粒子位置与速度
            if (pso_model == 'standard_pso'):
                x, v = self.update_standard_pso(x, v)
            elif (pso_model == 'ispsowm'):
                if ((self.step_max - step) < 10) or ((self.step_max - step) < 0.05 * self.step_max):
                    x, v = self.update_standard_pso(x, v)
                else:
                    x, v = self.update_ispsowm(x, v, step, c, pm, xi_wm, g)

            # 输出当前寻优进度--进度条形式
            '''
            time.sleep(0.1)
            
            if (step % 10) == 0:
                print(step,end='\r')
            '''
            self.progress_bar(step,self.step_max)

        # 若寻优达到预设最大代数且尚未收敛，则输出以下信息
        if (step == (self.step_max - 1)):
            print('已达到最大寻优代数','\n')

        _constraint_value_all = [constraint[index_f_gbest][i][1]
                             for i in range(len(constraint[index_f_gbest]))]

        # 判断寻优结果是否违反约束
        if (self._constraint_judge == True):
            # 获得最优解违反约束情况
            for i in range(len(_vio_cons_par)):
                if (index_f_gbest == _vio_cons_par[i][0]):
                    _constraint_violate_index = _vio_cons_par[i][1]
            if (penalty_type == 'oracle') and (update_oracle == True):
                _constraint_violate_index = []
            if (len(_constraint_violate_index) == 0):
                print('寻优结果满足约束要求','\n',
                      '结果函数值为', self.f_value,'\n',
                      '解向量为', self.x_gbest_step,'\n',
                      '约束函数值为', _constraint_value_all,'\n')
            else:
                print('寻优结果有变量不满足约束要求','\n',
                      '结果函数值为',self.f_value,'\n',
                      '解向量为', self.x_gbest_step,'\n',
                      '约束函数值为', _constraint_value_all,'\n')
                print('违反的约束索引号为：', _constraint_violate_index,'\n')
        elif (self._constraint_judge == False):
            print('寻优结果函数值为', self.f_value,'\n',
                  '解向量为', self.x_gbest_step,'\n')

        _constraint_info = [_constraint_violate_index,_constraint_value_all]

        return self.f_value, self.x_gbest_step,\
               _constraint_info, self.f_history,\
               self.feasible_f, self.feasible_x

    ## 将寻优过程全局最优函数值的变化过程可视化，调试时用
    def visualize(self, _f_history, fig_type='scatter'):
        '''
        将寻优过程全局最优函数值的变化过程可视化
        :param _f_history: 寻优过程全局最优函数值历史值集合，数组（浮点数）
        :param fig_type:   画图方式，取值为{'scatter'(散点图)(建议方式)，'plot'(折线图)}
        :return: 寻优过程全局最优函数值随寻优代数变化图
        '''
        _x_axis = []
        _x_axis.extend([i for i in range(len(_f_history))])

        if (fig_type == 'scatter'):
            plt.scatter(_x_axis, _f_history)
        elif (fig_type == 'plot'):
            plt.plot(_x_axis, _f_history)

        plt.xlabel("step")
        plt.ylabel("f")
        plt.title("f with step")

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

    path.append('..\\..\\..\\')
    from gensbo.gensbo import GenSBO
    from gensbo.core import Problem
    from gensbo.simulators.userfunction import UserFunction
    from gensbo.optimizers.pso import PSO
    import numpy as np
    from time import time

    to = time()

    INFINITY = 1e16  # 无穷大
    ## 创建优化模型信息
    #        全局最优： f(-3.1302468,-1.5821422) = -106.7645367
    problem = Problem("Mishra's Bird")

    # 添加变量
    problem.add_var('x1', "continuous", lowbound=-10, upbound=0, value=-3)
    problem.add_var('x2', "continuous", lowbound=-6.5, upbound=0, value=-1.6)

    problem._TotalNumVar = 2

    problem._TotalNumConstraint = 1

    problem._NumObjFunc = 1


    # 添加目标函数和约束函数
    def problem_function(varset):
        """
        添加目标函数和约束函数
        :param varset: 变量集,字典（'var_name':value）
        :return: 目标函数值list、约束值list，参考值flag
        """
        objfunc = []
        objfunc.extend([] for _ in range(problem._NumObjFunc))
        constraint = []
        constraint.extend([] for _ in range(problem._TotalNumConstraint))

        # 给变量名赋值（x1 = value)
        globals().update(varset)

        # 添加目标函数

        objfunc[0] = np.sin(x2) * np.exp((1 - np.cos(x1)) ** 2) + np.cos(
            x1) * np.exp((1 - np.sin(x2)) ** 2) + (x1 - x2) ** 2

        # 添加约束函数

        constraint[0] = ['i', (x1 + 5) ** 2 + (x2 + 5) ** 2 - 25]

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

    # 设置优化器运行参数，未改变的均为默认值
    optimizer.set_options('swarm_size', 30)
    optimizer.set_options('pso_mode', 'standard_pso')
    optimizer.set_options('penalty_type', 'common')
    #optimizer.set_options('oracle', -100)
    optimizer.set_options('precision', 3)
    optimizer.set_options('step_max', 100)
    optimizer.set_options('neighborhood', 'star')
    optimizer.set_options('if_mp',False)
    optimizer.set_options('mp_core_num',4)

    # print('para',optimizer.options)

    # 设置优化问题，主要是名字，目标函数
    # 查看变量集信息get_varset()
    # test = problem.get_varset()

    # print(problem)
    # optimizer.setoptions() #添加仿真器的结果收集器即可

    # 执行主程序
    gensbo = GenSBO(problem, simulator, optimizer)
    gensbo.run()

    # 获取寻优结果
    result = gensbo.result
    print(result)

    print('time',time()-to)
