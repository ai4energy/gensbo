# gensbo—a **GEN**eral **S**imulation **B**ased **O**ptimizer

gensbo的目标是创建一个基于仿真的通用优化器。

目前已实现基于改进简化小波粒子群算法的单目标优化和基于改进小波全面学习粒子群算法的多目标优化引擎。支持混合整数非线性优化问题，可以便捷自定义目标函数。

关于仿真器（如trnsys, energyplus）的调用模块（用于提供解的评价和约束信息，即目标函数值和约束函数值）尚未完成调试。

详情见gensbo说明文档（/doc）。



Notes: 重在实现功能，代码质量敬请谅解。



## 安装

```python
# pip 安装
pip install gensbo

# 下载代码
python setup.py install
```



## 示例

```python
# 单目标优化
/examples/user_function/MINLP_1.py
/examples/user_function/Mishra's Bird.py

# 多目标优化
/examples/user_function/B&K_2d.py

# 结果保存：自动生成3份文件
# 单目标优化（so）：
# 	name_so.npy：以np.array格式保存单目标优化（so）结果:[result,result_name]
#		result:[f, x, constraint_info， f_history, feasible_f, feasible_x]
#			f：目标函数寻优结果（浮点数）
#			x：寻优结果解向量（一维数组）
#			constraint_info：[violate_cons,cons_value_all]：[解向量违反的约束序号数组，解向量所有约束函数的值],
#			f_history：目标函数寻优历史（一维数组）
#			feasible_f:可行解的函数值集合（一维数组）
#			feasible_x:可行解集合（二维数组）
# 	name_view_so.xls：以xls格式保存单目标优化（so）结果
#		f：目标函数寻优结果
#		x：寻优结果解向量
#		cons_value_all:解向量所有约束函数的值
#		violate_cons：解向量违反的约束序号数组
#		f_history：目标函数寻优历史
#		feasible_f:可行解的函数值集合（一维数组）
#		feasible_x:可行解集合（二维数组）
# 	name_so.png：寻优历史图
# 多目标优化（mo）：
# 	name_mo.npy：以np.array格式保存多目标优化（mo）结果:[result,result_name]
#		result:[f_pareto, x_pareto, feasible_f, feasible_x]
#			f_pareto：寻优结果非支配解集（pareto前沿）的目标函数值集合（二维数组）,        
#			x_pareto：寻优结果非支配解向量（二维数组）
#			feasible_f:可行解的函数值集合（二维数组）
#			feasible_x:可行解集合（二维数组）
# 	name_view_mo.xls：以xls格式保存多目标优化（mo）结果
#		pareto_f1,pareto_f2 ...：寻优结果帕累托最优前沿目标函数值
#		x1,x2 ...：寻优结果解向量
#		feasible_f:可行解的函数值集合（二维数组）
#		feasible_x:可行解集合（二维数组）
# 	name_mo.png：寻优结果帕累托最优前沿目标函数值(f1-x轴；f2-y轴;f3-z轴)（只支持2D/3D）
```



## 导入gensbo模块

```python
from gensbo.gensbo import GenSBO
from gensbo.core import Problem
from gensbo.simulators.userfunction import UserFunction
from gensbo.optimizers.pso import PSO

import numpy as np
```



## 支持变量类型

变量类型：
	"continuous"：			  连续变量（浮点数）
	"discrete"：					连续离散变量（整数）
	"discrete_disconti"：	非连续离散变量（浮点数（取值集合））
	"binary"：					   二元变量（整数）



变量添加方法示例：
```python
# 创建问题实例
problem = Problem("function_name")

# 总变量数
problem._TotalNumVar = 5
# 总约束数
problem._TotalNumConstraint = 3
# 总目标函数数
problem._NumObjFunc = 1	#1：单目标优化（sopso）;  >=2：多目标优化（mopso）

# 添加变量
problem.add_var("x1", "continuous", 		lowbound=-15.0, 	upbound=15.0, 	value=None)
problem.add_var("x2", "discrete", 			lowbound=-5.0, 		upbound=9.0, 	value=4)
problem.add_var("x3", "discrete_disconti", 	set=[-5, -3, 0, 6, 9, 23], 			value=None)
problem.add_var("x4", "binary", 			lowbound=0, 		upbound=1, 		value=0)
# 初值value的选取应尽可能有助于提供最优解的信息，否则应将其设为‘None’（建议均设为‘None’）
# 使用PSO优化算法时，受到目标函数形态的影响，有时提供部分初值反而会诱使算法过早陷入局部最优，影响算法性能，请谨慎使用
# 在多目标优化中，通过设置某已知非支配解作为初值，同时减小寻优代数（‘step_max’）或增大外部档案容量（‘pareto_size’），可以在较短时间内获得该解周围的详细非支配解信息。

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

# 批量传入变量初始值
# 支持 list 和 np.array 格式
# 每个解向量初值中变量顺序应与添加变量顺序一致（与寻优结果导出的解向量中变量顺序相同）
if_batch_add_var_ini = True
if if_batch_add_var_ini == True:
    # 加载上次寻优导出的可行解
    var_ini = np.load('%s_so.npy'%name,allow_pickle=True)[0][-1]
    problem.batch_add_var_ini(var_ini)
```

## 优化问题定义（目标函数和约束函数）

```python

# 添加目标函数和约束函数
def problem_function(varset,if_cal_cons_only=False):
    """
    添加目标函数和约束函数
    :param varset: 变量集,字典（'var_name':value）
    :param if_cal_cons_only：布尔值，是否只计算约束函数值而不计算评价函数值，用于产生可行解
    :return: 目标函数值list、约束值list，参考值flag
    """
    objfunc = [0 for _ in range(problem._NumObjFunc)]
    constraint = [[] for _ in range(problem._TotalNumConstraint)]

    # 给变量名赋值（x1 = value)
    globals().update(varset)

    if if_cal_cons_only == False:
        # 添加目标函数
        objfunc[0] = -(- x1 * np.cos(np.pi * x1) + x1 * x2 + 4 * x3 - 5 * x4) - x5

    # 添加约束函数
    ######################################################
    # 第i个约束条件：constraint[i] = [cons_type,cons_value]
    # 约束类型：cons_type=['i','e']
    # 约束函数计算值：cons_value=[g(x),h[x]]
    # 'i'：不等式约束，形式为g(x)<=0
    # 'e'：等式约束，形式为|h(x)|-ϵ <= 0, ϵ默认值为1e-5。
    ######################################################
    constraint[0] = ['i', (x1 + x2) - 50]
    constraint[1] = ['i', (x1 + x1 * x3) - 34]
    constraint[2] = ['i', (x4 + x3) - 33.3]

    # 参考信息
    flag = 0
    return objfunc, constraint, flag
```



## PSO优化算法参数设置

```python
参数设置方式：
	# 创建优化器实例
	optimizer = PSO(problem)

	# 设置参数
	optimizer.set_options('para_name', value)

公用参数：

	'if_ini_cons': False,
        # 是否要求初始化的粒子必须含有可行解（不违反约束）,False：不要求；True:要求。

	'swarm_size': 30,  
		# 粒子数量
	'step_max': 500,
		# 最大寻优代数
	'w_type':'linear',
		# 惯性权重方式，'linear'-线性，'fixed'-定值，'chaos'-混沌惯性权重
		
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
	
单目标粒子群算法sopso：

	'pso_mode': "ispsowm",
		# 选择用于运算的单目标pso方法，字符串（default：'ispsowm'）
		# 目前提供：
		# "standard_pso'-标准粒子群算法
		# 'ispsowm'-改进小波简化粒子群算法（复杂问题推荐）
		# 选择ispsowm时，
		# c（反向搜索阈值）, pm（变异阈值）, xi_wm（形状因子）,g（函数a的上限值） 参数生效。
		# 若搜索过程表明粒子群明显陷入早熟，则适当增大c，pm的值。具体参数设置请参考说明文档。
	
	'w': 0.8,
	# 惯性权重（default：0.8）
	'w_range_so': [0.4, 1.2],
	# 惯性权重变化范围，数组（浮点数）（default：[0.4, 1.2]）
	'c1': 2.0,
	# 加速因子中的认知因子（default：2）
	'c2': 2.0,
	# 加速因子中的社会因子（default：2）
	
	'precision': 0.001,
	# 收敛精度，浮点数（default：1e-3）
	# 寻优程序设定当种群粒子的个体历史最好目标函数值（集合）的标准差小于收敛精度时收敛。
	# 问题规模较小时，可将precision调大
	
	'neighborhood': 'star',
	# 粒子群邻域结构，'star'（全互连型），'ring'（环型）（搜索速度较慢，但更不容易陷入早熟）	

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
```


​	多目标粒子群算法mopso：
​	

```python
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
```

## 运行寻优模块

```python
    # 执行主程序
    gensbo = GenSBO(problem, simulator, optimizer)

    # 记录当前寻优结果，防止意外报错使得进程结束--代码有误
    result_temp = gensbo.result_temp

    # 记录当前寻优可行解结果，防止意外报错使得进程结束
    feasible_f_temp = gensbo.feasible_f_temp
    feasible_x_temp = gensbo.feasible_x_temp

    gensbo.run()

    # 获取寻优结果
    result = gensbo.result
    '''
    单目标优化结果
    result = [f, x, constraint_info， f_history, feasible_f, feasible_x]
               [寻优结果函数值（浮点数），寻优结果解向量（一维数组），
                [解向量违反的约束序号数组，解向量所有约束函数的值],
                寻优历史全局最优函数值（一维数组），
                可行解的函数值集合（一维数组），可行解集合（二维数组）]
    
    
    多目标优化结果
    result = [f_pareto, x_pareto, feasible_f, feasible_x]
               [寻优结果非支配解集（pareto前沿）的目标函数值集合（数组（浮点数））,
                寻优结果非支配解向量（数组（浮点数））,
                可行解的函数值集合（二维数组），可行解集合（二维数组）]
    '''

    # 保存数据
    gensbo.save_data(filename=problem.name, filepath='.\\')

    # 结果可视化，若需保存图片则需输入文件名及保存文件路径
    gensbo.visualize(filename=problem.name, filepath='.\\')
```



## Benchmark函数测试

- 测试环境：
  - 系统：Windows 7 64位操作系统
  - 处理器：Intel(R) Core(TM) i7-6700 CPU @ 3.40 GHz
  - 安装内存（RAM）: 8.00 GB

### Rastrigin函数（30维，多峰，单目标函数）

- 全局最优：$f(0,…,0)=0$

- 测试结果：

  | **算法**  **（相同计算条件）** | **标准粒子群算法（standard_pso）** | 简化小波粒子群算法（ispsowm） |
  | ------------------------------ | ---------------------------------- | ----------------------------- |
  | 50次计算平均值                 | 146.01                             | 2.99×10-4                     |
  | 50次计算最大值                 | 201.23                             | 9.77×10-4                     |
  | 单次计算平均耗时               | 0.21  s                            | 0.023  s                      |
  | 平均寻优代数                   | 100（设置上限）                    | 7.86                          |
  | 备注                           | 粒子数量：30，收敛精度：10-3       | 粒子数量：30，收敛精度：10-3  |

### ZDT-1 （目标函数：2）

- 世代距离（GD）：$9.29 \times 10^{-4}$

  ![image-20210112170739174](https://github.com/ai4energy/gensbo/blob/main/README.assets/image-20210112170739174.png)

### Viennet3 function（目标函数：3）

- 世代距离（GD）：$2.06 \times 10^{-3}$

  ![image-20210112170903075](../README.assets/image-20210112170903075.png)



