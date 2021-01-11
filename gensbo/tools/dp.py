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
version-0.1.200713_beta
改变：
    (1)结果输出内容细化
    (2)图片增加标题与坐标轴名称

文件说明：
    数据处理：数据输出，可视化

函数说明：


    autorun()       ## 自动执行寻优结果可视化和保存数据操作

    transform_f(f_pareto)
    ## 转换格式：f[i][j]:第i个解（粒子）的第[j]个目标函数值——_f[i][j]:第i个目标函数数列中对应第j个解（粒子）对应的值

    visualize()
    ## 可视化：单目标优化-优过程全局最优函数值的变化过程，多目标优化-寻优结果pareto前沿（目标函数个数为2或3）

    write_to_file()                     ## 数据输出保存

'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xlwt

## 自动执行寻优结果可视化和保存数据操作
def autorun(algo_name, result, filename, savefig= True, showfig= False, fig_type= 'scatter'):
    '''
    自动执行寻优结果可视化和保存数据操作
    :return:
    '''

    # 可视化
    visualize(algo_name, result, filename, savefig= savefig, showfig= showfig, fig_type= fig_type)

    # 保存数据到文件
    write_to_file(algo_name, result, filename)

## 可视化：单目标优化-优过程全局最优函数值的变化过程，多目标优化-寻优结果pareto前沿（目标函数个数为2或3）
def visualize(algo_name, result, filename, savefig= True, showfig= False, fig_type= 'scatter'):
    '''
    可视化：单目标优化-优过程全局最优函数值的变化过程，多目标优化-寻优结果pareto前沿（目标函数个数为2或3）
    :return: 图
    '''
    if (algo_name == 'sopso'):
        _f_history = result[3]
        _x_axis = []
        _x_axis.extend([i for i in range(len(_f_history))])

        if (fig_type == 'scatter'):
            plt.scatter(_x_axis, _f_history)
        elif (fig_type == 'plot'):
            plt.plot(_x_axis, _f_history)

        plt.xlabel("step")
        plt.ylabel("f")
        plt.title("f_gbest with step")

    elif (algo_name == 'mopso'):
        # 转换格式
        _f_pareto = transform_f(result[0])

        fig = plt.figure()
        # 二维画图
        if (len(_f_pareto) == 2):
            if (fig_type == 'scatter'):
                plt.scatter(_f_pareto[0], _f_pareto[1])
            elif (fig_type == 'plot'):
                plt.plot(_f_pareto[0], _f_pareto[1])

            plt.xlabel("f1(x)")
            plt.ylabel("f2(x)")
            plt.title("Pareto Front")

        # 三维画图
        elif (len(_f_pareto) == 3):

            ax = Axes3D(fig)
            ax.scatter(_f_pareto[0], _f_pareto[1], _f_pareto[2])

            plt.title("Pareto Front")
            ax.set_xlabel('f1(x)')
            ax.set_ylabel('f2(x)')
            ax.set_zlabel('f3(x)')

    if (showfig == True):
        plt.show()

    if (savefig == True):
        if (algo_name == 'sopso'):
            plt.savefig(filename + '_so')
        elif (algo_name == 'mopso'):
            plt.savefig(filename + '_mo')

## 转换格式：f[i][j]:第i个解（粒子）的第[j]个目标函数值——_f[i][j]:第i个目标函数数列中对应第j个解（粒子）对应的值
def transform_f(f_pareto):
    '''
    转换格式：f[i][j]:第i个解（粒子）的第[j]个目标函数值——_f[i][j]:第i个目标函数数列中对应第j个解（粒子）对应的值
    针对mopso输出结果，用于可视化
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

## 数据输出保存
def write_to_file(algo_name, result, filename, var_name = []):
    '''
    数据输出保存(利用pandas实现）
    :return:
    '''
    if (algo_name == 'sopso'):
        np.save(filename + '_so.npy',[result,['f','x',
                                              'cons_info=[vio_cons_index,cons_value_all]',
                                              'f_history',
                                              'feasible_f',
                                              'feasible_x']])
        f_value,x_value,cons_info,f_history,fea_f,fea_x = result
        vio_cons_index,cons_value_all = cons_info
        # 创建便于直观查看结果的excel文档
        wb = xlwt.Workbook()
        ws = wb.add_sheet('result')
        ws.write(0, 0, 'f_value')
        ws.write(1, 0, f_value)
        ws.write(0, 1, 'x_name')
        ws.write(0, 2, 'x_value')
        if len(var_name) > 0:
            for i in range(len(x_value)):
                ws.write(1 + i, 1, var_name[i])
                ws.write(1+i, 2, x_value[i])
        else:
            for i in range(len(x_value)):
                ws.write(1 + i, 1, None)
                ws.write(1+i, 2, x_value[i])

        ws.write(0, 3, 'cons_value')
        for i in range(len(cons_value_all)):
            ws.write(1+i, 3, cons_value_all[i])

        ws.write(0, 4, 'violate_cons')
        if (len(vio_cons_index) == 0):
            ws.write(1, 4, 'None')
        else:
            for i in range(len(vio_cons_index)):
                ws.write(1+i, 4, vio_cons_index[i])

        ws.write(0, 5, 'f_history')
        for i in range(len(f_history)):
            ws.write(1+i, 5, f_history[i])

        if len(fea_f) > 0:
            ws.write(0, 6, 'feasible_f')
            ws.write(0, 7, 'feasible_x')
            for i in range(len(fea_f)):
                ws.write(1+i, 6, fea_f[i])
                ws.write(1 + i, 7, str(fea_x[i]))

        wb.save(filename + '_view_so.xls')

    elif (algo_name == 'mopso'):
        np.save(filename + '_mo.npy',[result,['f_pareto', 'x_pareto','feasible_f','feasible_x']])

        # 创建便于直观查看结果的excel文档
        wb = xlwt.Workbook()
        ws = wb.add_sheet('result')
        f_pareto = result[0]    # 目标函数个数
        ob_number = len(f_pareto[0])
        x_pareto = result[1]    # 变量个数
        x_number = len(x_pareto[0])
        pareto_number = len(f_pareto)
        feasible_f = result[2]
        feasible_x = result[3]
        feasible_number = len(feasible_f)

        # pareto前沿目标函数值
        for i in range(ob_number):
            ws.write(0, i, 'pareto_f%s'%(i+1))
            for j in range(pareto_number):
                ws.write(j+1, i, f_pareto[j][i])

        # pareto前沿解向量值
        for i in range(ob_number,ob_number+x_number):
            ws.write(0, i, var_name[i-ob_number])
            for j in range(pareto_number):
                ws.write(j+1, i, x_pareto[j][i-ob_number])

        # 可行解
        # 目标函数值：list
        ws.write(0, ob_number+x_number, 'feasible_f')
        for i in range(feasible_number):
            ws.write(i+1,ob_number+x_number,str(feasible_f[i]))

        # 可行解向量:list
        ws.write(0, ob_number+x_number+1, 'feasible_x')
        for i in range(feasible_number):
            ws.write(i+1,ob_number+x_number+1,str(feasible_x[i]))

        wb.save(filename + '_view_mo.xls')




