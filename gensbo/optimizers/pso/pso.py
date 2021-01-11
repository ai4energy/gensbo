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
path.append('..\\..\\..\\')
from gensbo.optimizers.pso import sopso
from gensbo.optimizers.pso import mopso
from gensbo.optimizers.optimizer import BaseOptimizer
from gensbo.core.problem import Problem
from gensbo.simulators.simulator import BaseSimulator



class PSO(BaseOptimizer):
    name = "PSO"

    def __init__(self, problem, *args, **kwargs):

        # 设置问题（实例）
        self.problem = problem

        # 记录当前寻优结果，防止意外报错使得进程结束
        self.result_temp = []

        # 记录寻优过程产生的可行解及其评价函数值
        self.feasible_x_temp = []
        self.feasible_f_temp = []

        """优化问题"""
        # 问题必须是Problem实例
        # 仿真器必须是实例simulator
        # 默认使用标准的pso算法
        # self.psomode = "standardpso"
        # self.psomode = "ipsowm"
        # """模式，标准粒子群？小波？量子粒子群？"""

        # self.swarm_size = None
        # """粒子群大小"""

        # self.inertialweight = None
        # """惯性权重"""

        # self.c1 = None
        # """认知因子"""

        # self.c2 = None
        # """社会因子"""

        # self.step_max = None
        # """最大步数"""

        # self.precision = None
        # """收敛精度"""

        default_options = {
            'pso_mode': "ispsowm",
            # 选择用于运算的pso方法，字符串（default：'ispsowm'）
            # 目前提供：
            # "standard_pso'-标准粒子群算法
            # 'ispsowm'-改进小波简化粒子群算法（推荐）
            # 选择ispsowm时，
            # c（反向搜索阈值）, pm（变异阈值）, xi_wm（形状因子）,g（函数a的上限值） 参数生效。
            # 若搜索过程表明粒子群明显陷入早熟，则适当增大c，pm的值。具体参数设置请参考说明文档。
            'if_ini_cons': False,
            # 是否要求初始化的粒子必须含有可行解（不违反约束）,False：不要求；True:要求。
            'swarm_size': 30,
            'w': 0.8,
            # 惯性权重（default：0.8）
            'c1': 2.0,
            # 加速因子中的认知因子（default：2）
            'c2': 2.0,
            # 加速因子中的社会因子（default：2）
            'step_max': 500,
            'verbose': 1,
            'precision': 0.001,
            # 收敛精度，浮点数（default：1e-3）
            # 寻优程序设定当种群粒子的个体历史最好目标函数值（集合）的标准差小于收敛精度时收敛。
            # 问题规模较小时，可将precision调大

            # 许可误差精度(accuracy of constraint violation):用于等式约束
            'acc_cons_vio':1e-5,

            'if_mp':False,# 是否使用并行计算（多进程），默认不使用
            'mp_core_num':2,# 并行计算时使用的cpu核数，默认为2

            # 是否初始化内点（可行域）
            'if_ini_cons':False,
            # 内点个数
            'ini_feasible_x_num':1,
            # 初始化内点的最大运算次数
            'ini_step':30,
            # 初始化内点中sopso的参数设置
            'ini_swarm_size':50,
            'ini_step_max':1000,
            'ini_precision':1e-3,

            # 是否收集寻优过程产生的可行解及其适应度函数值
            'if_get_feasible_x':False,
            # 当可行解个数满足要求时是否停止寻优：只寻找非重复的可行解
            'if_get_feasible_x_only':False,
            # 记录可行解个数上限（整数）
            'feasible_x_num':100,

            'w_type':'linear',
            'w_range_so': [0.4, 1.2],
            # 惯性权重变化范围，数组（浮点数）（default：[0.4, 1.2]）
            'neighborhood': 'star',
            # 粒子群邻域结构，'star'（全互连型），'ring'（环型）（搜索速度较慢，但更不容易陷入早熟）

            # {{{惩罚函数相关参数
            'penalty_type': 'common',
            # 选择罚函数方法，字符串（default：'common'）
            # 'common' - 普通（动态）罚函数，'oracle' - oracle罚函数
            # 粒子群多目标优化算法不支持oracle罚函数方法
            'penalty_times': 100,
            # penalty_type = 'common'时生效，惩罚倍数，浮点数（default：100）
            # 使违反约束的解受到惩罚后的函数值一定大于全局最优适应函数值
            'oracle': 1e9,
            # penalty_type = 'oracle'时生效，Ω初始值，浮点数（default：1e9）
            # 该值必须大于全局最优适应函数值
            # }}}惩罚函数

            # {{{小波相关参数
            'c': 0.2,
            # 反向搜索概率，浮点数（default：0.2）
            'pm': 0.7,

            # 执行小波变异的概率阈值，浮点数（default：0.7），取值范围[0,1]
            'xi_wm': 0.5,
            # 形状参数，浮点数（default：0.5）
            'g': 1000,
            # 小波函数中a的上限值，浮点数（default：1000），常取1000或10000
            # }}}小波

            # {{{ 多目标相关参数
            'x_pareto_size': 100,
            # 外部档案大小（存放非支配解），整数（default：100）
            'if_use_former_x':True,
            # 是否利用前次寻优得到的pareto前沿（稀疏部分）引导本次寻优
            'w_mo': 0.4,
            # 惯性权重（default：0.4）
            'w_range_mo': [0.2, 0.6],
            # 惯性权重变化范围，数组（浮点数）（default：[0.2, 0.6]）
            'c_mo': 2.0,
            # 加速因子，浮点数（default：2）
            'run_number': 5,
            # 独立寻优次数，整数（default：5），合并各次寻优结果pareto前沿
            'pm_mo': 0.7,
            # 执行小波变异的概率阈值，浮点数（default：0.7），取值范围[0,1]
            'xi_wm_mo': 0.5,
            # 形状参数，浮点数（default：0.5）
            'g_mo': 1000,
            # 小波函数中a的上限值，浮点数（default：1000），常取1000或10000
            'pe_mo': 0.4,
            # 精英概率，浮点数（default：0.4），取值范围[0,1]
            'pl_mo': 0.1,
            # 学习概率，浮点数（default：0.1），取值范围[0,1]
            # }}}多目标相关参数
        }

        self._options = default_options

    '''
    def pso_randomswarm_generator(self):
        """粒子群产生器"""
        pass

    def pso_pbestupdate(self):
        """找到并更新当前代的最优"""
        pass

    def pso_gbestupdate(self):
        """找到并更新全局最优"""
        pass

    def findbestneighbor(self):
        pass

    def pso_updateposition(self):
        """更新粒子的位置"""
        pass

    def pso_updatevelocity(self):
        """更新粒子的速度"""
        pass

    def _run(self, problem):
        """根据问题开始产生一组粒子，如果是step不为1则，根据当前粒子的评价结果产生一组粒子"""

        return "hello"
    
    def setproblem(self, problem):
        if not (isinstance(problem, Problem)):
            raise ValueError("参数必须是Problem类的实例")
        else:
            self.problem = problem
        # end

    '''


    def set_simulator(self, simulator):
        """
        设置仿真器（实例）
        :param simulator:
        :return:
        """
        if not (isinstance(simulator, BaseSimulator)):
            raise ValueError("参数必须是BaseSimulator子类的实例")
        else:
            self.simulator = simulator
        # end

    def __str__(self):
        # TODO
        pass

    def __repr__(self):
        pass

    @property
    def options(self):
        return self._options

    def set_options(self, key, value):
        """设置pso优化器选项"""
        if key not in self.options:
            raise AttributeError('unknown option key: ' + str(key))
        self.options.__setitem__(key, value)

    def opt(self):
        """
        调用算法寻优
        :return:result: 寻优结果
        """
        if (self.problem._NumObjFunc == 1):
            self.name = 'sopso'
            # 变量信息调用problem，目标函数和约束信息调用simulator
            algo = sopso.Algorithm(self.problem, self.simulator, swarm_size=self.options['swarm_size'],
                                   w=self.options['w'], c1=self.options['c1'], c2=self.options['c2'],
                                   step_max=self.options['step_max'], precision=self.options['precision'],
                                   if_mp=self.options['if_mp'], mp_core_num=self.options['mp_core_num'],
                                   if_ini_cons=self.options['if_ini_cons'],
                                   ini_feasible_x_num=self.options['ini_feasible_x_num'],
                                   acc_cons_vio=self.options['acc_cons_vio'],
                                   ini_step=self.options['ini_step'],
                                   if_get_feasible_x=self.options['if_get_feasible_x'],
                                   if_get_feasible_x_only=self.options['if_get_feasible_x_only'],
                                   feasible_x_num=self.options['feasible_x_num'],
                                   ini_swarm_size=self.options['ini_swarm_size'],
                                   ini_step_max=self.options['ini_step_max'],
                                   ini_precision=self.options['ini_precision'])

            # 记录当前寻优结果，防止意外报错使得进程结束
            self.result_temp.append(algo.result_temp)

            # 记录寻优过程产生的可行解及其评价函数值
            self.feasible_x_temp.append(algo.feasible_x_temp)
            self.feasible_f_temp.append(algo.feasible_f_temp)

            ## 选择求解算法
            result = algo.run(pso_model=self.options['pso_mode'],
                              neighborhood=self.options['neighborhood'],
                              c=self.options['c'], pm=self.options['pm'], xi_wm=self.options['xi_wm'],
                              g=self.options['g'], penalty_type=self.options['penalty_type'],
                              penalty_times=self.options['penalty_times'], oracle=self.options['oracle'],
                              w_type=self.options['w_type'], w_range=self.options['w_range_so'])
            '''
            result = [f, x, constraint_info， f_history, feasible_f, feasible_x] 
                     [寻优结果函数值（浮点数），寻优结果解向量（一维数组），
                      [解向量违反的约束序号数组，解向量所有约束函数的值],
                      寻优历史全局最优函数值（一维数组），
                      可行解的函数值集合（一维数组），可行解集合（二维数组）]
            '''

        ## 多目标优化
        elif (self.problem._NumObjFunc > 1):
            self.name = 'mopso'
            ## 设置求解器参数
            algo = mopso.Algorithm(self.problem, self.simulator, swarm_size=self.options['swarm_size'],
                                   w_mo=self.options['w_mo'], c_mo=self.options['c_mo'],
                                   step_max=self.options['step_max'], x_pareto_size=self.options['x_pareto_size'],
                                   if_mp=self.options['if_mp'], mp_core_num=self.options['mp_core_num'],
                                   if_ini_cons=self.options['if_ini_cons'],
                                   ini_feasible_x_num=self.options['ini_feasible_x_num'],
                                   acc_cons_vio=self.options['acc_cons_vio'],
                                   ini_step=self.options['ini_step'],
                                   if_get_feasible_x=self.options['if_get_feasible_x'],
                                   if_get_feasible_x_only=self.options['if_get_feasible_x_only'],
                                   feasible_x_num=self.options['feasible_x_num'],
                                   ini_swarm_size=self.options['ini_swarm_size'],
                                   ini_step_max=self.options['ini_step_max'],
                                   ini_precision=self.options['ini_precision'])

            # 记录当前寻优结果，防止意外报错使得进程结束
            self.result_temp.append(algo.result_temp)

            # 记录寻优过程产生的可行解及其评价函数值
            self.feasible_x_temp.append(algo.feasible_x_temp)
            self.feasible_f_temp.append(algo.feasible_f_temp)

            ## 选择求解算法
            # 进行多次计算合并结果，以得到更全面的pareto前沿
            result = algo.run_repeat(run_number=self.options['run_number'], if_ini_cons=self.options['if_ini_cons'],
                                     pm_mo=self.options['pm_mo'],
                                     xi_wm_mo=self.options['xi_wm_mo'], g_mo=self.options['g_mo'],
                                     pe_mo=self.options['pe_mo'], pl_mo=self.options['pl_mo'],
                                     penalty_type=self.options['penalty_type'],
                                     penalty_times=self.options['penalty_times'],
                                     w_type=self.options['w_type'], w_range=self.options['w_range_mo'],
                                     if_use_former_x=self.options['if_use_former_x'])

            '''
            result = [f_pareto, x_pareto, feasible_f, feasible_x]
                     [寻优结果非支配解集（pareto前沿）的目标函数值集合（数组（浮点数））,
                      寻优结果非支配解向量（数组（浮点数））,
                      可行解的函数值集合（二维数组），可行解集合（二维数组）]
            '''

        return result

if __name__ == "__main__":
    """测试"""
    print("Testing ...")

    from gensbo.core import Problem
    from gensbo.simulators import UserFunction

    prob = Problem("testing problem")
    # 创建一个问题并设置相应的

    prob._TotalNumVar = 5
    prob._NumBinVar = 0
    prob._NumIntVar = 2
    prob._NumContinusVar = 3
    prob.add_var("x1")
    prob.add_var("x2")
    prob.add_var("x3")

    print(prob)

    # 创建一个优化器，并设置选项
    optimizer = PSO()

    optimizer.set_options("inertialweight", 0.8)
    optimizer.set_options("c1", 2.0)
    optimizer.set_options("c2", 2.0)
    optimizer.set_options("precision", 0.0001)

    optimizer.set_problem(prob)

    print(optimizer.options)

    # 创建一个仿真器，并设置选项
    simulator = UserFunction()

    # 通过simulator将目标函数、约束函数传入

    from gensbo.tests.test import problem_function
    simulator.setobjfunc(problem_function)

    # 本质上，我们调用PSO优化器的_run方法是产生一组粒子
    # 注意，我们这里并没有使用GenSBO这个帽子类的实例
    # 这里_run方法需要一个参数problem，会根据

    optimizer.set_simulator(simulator)

    particles = optimizer._run(prob)

    # 然后调用仿真器评价这些粒子，需要注意的是这里的pariticles是多个粒子，我们simulate评价的是一个粒子

    # 对于UserFunction类型的仿真器，仿真的时候就会调用其problem_function获得评价和约束
    functionvalue = simulator.simulate(particles)
