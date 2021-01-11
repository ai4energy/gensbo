# gensbo使用手册

自定义函数示例代码文件位于 example/user_function

​	单目标：MINLP_1.py，Mishra's Bird.py

​	多目标：B&K_2d.py



变量类型：
	"continuous"：			连续变量（浮点数）
	"discrete"：				连续离散变量（整数）
	"discrete_disconti"：	非连续离散变量（浮点数（取值集合））
	"binary"：				二元变量（整数）

变量添加方法示例：
	# 创建问题实例
	problem = Problem("function_name")
	
```python
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



PSO优化算法所有参数：

```python
参数设置方式：
	# 创建优化器实例
	optimizer = PSO(problem)

	# 设置参数
	optimizer.set_options('para_name', value)

公用参数：

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
	
	# 并行计算：建议当目标函数的计算比较耗时的时候才启用，
	# 否则不同cpu核间的通讯消耗足以抵消并行计算的优势
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
    'feasible_x_num':100

	
单目标pso：

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

​	多目标pso：
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

