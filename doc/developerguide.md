# gensbo开发者手册

# 简介

# 类结构框架组织

Problem类——优化问题类，在gensbo.core.problem，使用了Varaible，objfunc，constraint类

BaseSimulator类——仿真器基类，包含有PSO类，这是我们目前实现的主要算法

BaseOptimizer类——优化器基类，包含有energyplus，trnsys，userfunction等，首先实现的是userfunction

GenSBO——我们的“帽子”类，抓有全部的handle，便于简化问题处理。

tools目录下是要使用的一些公共的工具。

所用package：
numpy
matplotlib
xlwt（数据处理，dp模块）

一、框架组织：
	1、三大模块：
		problem：问题类（Problem），包含变量，约束，目标函数信息（目前实现的自定义函数信息均由用户创建）
		simulator：仿真器类（UersFunction，Trnsys等），问题为自定义函数时simulator调用Problem实例中的目标函数和约束函数进行解的评价，仿真器时调用仿真器进行评价
		optimizer：优化器类（目前只有PSO），调用问题Problem实例的变量信息产生粒子，根据目标函数个数调用单/多目标pso算法，调用仿真器simulator实例进行解的评价
		
	2、寻优执行程序：GenSBO类
		产生GenSBO实例gensbo(problem, simulator, optimizer)，调用三大模块进行求解，并调用数据处理模块保存数据、可视化结果等
		
	3、算法包gensbo
		（1）gensbo组成
			core：				寻优模型
				problem.py：	寻优问题类
				variable.py:	变量类，将外界输入的变量形成变量字典以供使用
				__init__.py
	
			tools：				工具包
				dp.py：			寻优结果数据处理，画图，保存图片，保存数据到文件
				util.py:		打印信息（模板）
				__init__.py
		
			optimizers：		寻优算法
				optimizer.py:	优化器基类
				pso：			粒子群优化算法包
					pso.py:     PSO(BaseOptimizer):
								"""
								功能：
									1）传递寻优模型信息（函数setproblem（prob））和调用模型信息返回当前解的评价值（函数setsimulator（simulator））
									2) 传递优化器参数（函数set_options）；
									3）调用优化器进行寻优（函数opt()）
								"""
					mopso.py:	多目标粒子群算法（全面学习小波mopso）
					sopso.py:	单目标粒子群算法（标准pso，改进小波简化pso）
					__init__.py
				__init__.py
		
			simulators：			模拟器调用
				energyplus:
					base.py：	模拟器寻优模型建立，包括变量，目标函数（调用模拟器进行评价），约束函数
					__init__.py
				trnsys:
					base.py
					__init__.py
				__init__.py
				userfunction.py: UserFunction(BaseSimulator)
								"""
								功能：
									1）传递寻优模型函数信息（函数setobjfunc(problem_objfunc)）
									2) 调用模型函数信息返回当前解的目标函数值和约束函数值（函数simulate（varset））
										(f,con = simulator(实例).simulate(varset)[0:2])
								"""
			gensbo.py：			寻优执行主程序(GenSBO(object))
			_version_.py
			__init__.py
		
		（2）寻优运行方式
		gensbo中函数run()调用optimizer，
			optimizer调用问题Problem实例的变量信息产生粒子,
			optimizer调用仿真器simulator实例进行解的评价.
		其中，问题为自定义函数时，simulator调用Problem实例中的目标函数和约束函数进行解的评价，仿真器时则调用仿真器进行评价（具体文件组织模式待定）

二、运行示例：以自定义函数"MINLP_1"为例

```python
1、创建问题信息
	problem = Problem("MINLP_1")
	#全局最优f(15,35,1,-5,1) = -570
	
	# 总信息
	problem._TotalNumVar = 5
	problem._TotalNumConstraint = 3
	problem._NumObjFunc = 1
	
	# 添加变量
	problem.add_var("x1", "continuous", lowbound=-15.0, upbound=15.0, value=0)      # 连续变量
	problem.add_var("x2", "continuous", lowbound=0, upbound=100, value=50)			
	problem.add_var("x3", "discrete", lowbound=-15.0, upbound=15.0, value=0)		# 连续离散整型变量
	problem.add_var("x4", "discrete_disconti", set=[-5, -3, 0, 6, 9, 23], value=6)	# 非连续离散变量
	problem.add_var("x5", "binary", lowbound=0, upbound=1, value=0)					# 二元变量

	# 添加目标函数和约束函数
	def problem_function(varset,if_cal_cons_only=False):
		"""
		添加目标函数和约束函数
		:param varset: 变量集,字典（'var_name':value）
		:return: 目标函数值list、约束值list，参考值flag
		"""
		# 生成相应数组
        objfunc = [0 for _ in range(problem._NumObjFunc)]
        constraint = [[] for _ in range(problem._TotalNumConstraint)]

		# 给变量名赋值（x1 = value)
		globals().update(varset)

		# 添加目标函数
		if if_cal_cons_only == False:
			objfunc[0] = -(- x1 * np.cos(np.pi * x1) + x1 * x2 + 4 * x3 - 5 * x4) - x5

		# 添加约束函数
		constraint[0] = ['i', (x1 + x2) - 50]
		constraint[1] = ['i', (x1 + x1 * x3) - 34]
		constraint[2] = ['i', (x4 + x3) - 33.3]

		# 参考信息
		flag = 0
		
		return objfunc, constraint, flag

	# 将目标函数和约束函数加入问题实例，存于problem._function
	problem.add_objfunc(problem_function)

2、创建仿真器
	
	# 创建仿真器实例
	simulator = UserFunction()
	
	# 自定义函数调用优化问题Problem类实例problem的目标函数和约束函数信息
	simulator.set_objfunc(problem._function)
	# 用评价解的函数及形式 f,con = simulator.simulate(varset)

3、创建优化器

	# 创建优化器实例
	optimizer = PSO(problem)
	
	# 设置仿真器
	optimizer.set_simulator(simulator)

	# 设置优化器运行参数，未改变的均为默认值
	optimizer.set_options('pso_mode', 'standard_pso')
	optimizer.set_options('penalty_type', 'oracle')
	optimizer.set_options('precision', 0.01)
	#显示所有参数 print('para',optimizer.options)
```


```python
4、执行寻优
	
	# 创建寻优实例
	gensbo = GenSBO(problem, simulator, optimizer)
    
    # 记录当前寻优结果，防止意外报错使得进程结束--代码有误
    result_temp = gensbo.result_temp

    # 记录当前寻优可行解结果，防止意外报错使得进程结束
    feasible_f_temp = gensbo.feasible_f_temp
    feasible_x_temp = gensbo.feasible_x_temp
	
	# 执行寻优
	gensbo.run()

	# 获取寻优结果
	result = gensbo.result

	# 保存数据
	gensbo.save_data(filename=problem.name, filepath='.\\')

	# 结果可视化，若需保存图片则需输入文件名及保存文件路径
	gensbo.visualize(filename=problem.name, filepath='.\\')
```



​		