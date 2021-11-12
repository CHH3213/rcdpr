# -*-coding:utf-8-*-
'''
简单的测试程序
'''
import numpy as np

import matplotlib.pyplot as plt
import scipy.io as sio

# 测试列表相加
# list1=[1,2,3]
# list2 = [2,2,4]
# list3 = [5,6,7]
# # print(list1.append(1))
# # print(list1+list2)
# #
# list3 = np.array([np.array(list1), np.array(list2), np.array(list3)])
# print(list3[:,2])
# # print(list3)
# # print(np.shape(list3))
#
# # for i in range(0, 3, 1):
# #     print(i)
# # list2_ = np.array([[1],[1],[1]])
# list2_ = np.array([1,1,1])
# # list2_ = [1,1,1]
# print(np.shape(list2_))
# print(np.dot(np.linalg.pinv(list3), list2_))
# # list4 = np.linalg.solve(list3, list2_)
# print(np.linalg.solve(list3, list2_))
# list4_ = np.linalg.lstsq(list3,list2_)
# print(list4_)

# print(list4-list2_)
# print(type(list4-list2_))
# print(list3-list2_)
#
# cables_one_side = np.array([np.array([0,0,1])+np.array([[1,2,3],[4,5,6]])])  # 绳子一端
# cables_one_side = np.squeeze(cables_one_side)
# cables_other_side = np.array([[5,5,6],[9,8,7]])  # 绳子ling一端
# cable_lines = np.array([np.array([cables_one_side[i],cables_other_side[i]]) for i in range(len(cables_other_side))])
# print('==============')
# print(np.shape(cables_one_side))
# print(np.shape(cables_other_side))
# print(cables_one_side)
# print(cables_other_side)
# print(np.shape(cable_lines))
# print('+++++++++++++++++')
# cable_length = np.array([4.5 for _ in range(2)])
# print(cable_length)
# assert len(cable_length)==len(cable_lines)
# # flatten
# cable_lines = cable_lines.flatten()
# print(cable_lines)

# arr = np.array([np.array([1,2,3]) +  np.array([2, 5, 6])])
# arr = np.squeeze(arr)
# print(arr)
# print(np.shape(arr))

# from casadi import *
# # Symbols/expressions
# x = MX.sym('x')
# y = MX.sym('y')
# z = MX.sym('z')
# f = x**2+100*z**2
# g = z+(1-x)**2-y
#
# nlp = {}                 # NLP declaration
# nlp['x']= vertcat(x,y,z) # decision vars
# nlp['f'] = f             # objective
# nlp['g'] = g             # constraints
#
# # Create solver instance
# F = nlpsol('F','ipopt',nlp);
#
# # Solve the problem using a guess
# F(x0=[2.5,3.0,0.75],ubg=0,lbg=0)

# A = np.array([[1,2,3],[1,1,1]])
# a = []
# for i in range(2):
#     temp = np.array(A[i])/np.linalg.norm(A[i])
#     a.append(temp)
#     print(a[i])
#
# A = [1,2,2]
# B = [1,1,1]
# B =A
# print(B)
# print(A)

# a = np.array([[1,2,3],[2,3,4]])
# b = np.array([[1,2,3],[2,3,4],[3,4,4]])
# print(a@b)  # 矩阵乘法
#
#
# def funcA(A):
#     print("function A")
#     print(A)
#
#
# def funcB(B):
#     print(B(2))
#     print("function B")
#
#
# @funcA  # 这句话相当于funcA(test1),也就是把当前函数传过去
# def t1():
#     return "test1"
#
#
# # 函数和装饰器是倒着执行的，从下往上
# @funcA
# @funcB
# def func(c):
#     print("function C")
#     return c ** 2-1.732,-1,0

# a = np.array([[1,2,2],[2,3,4]])
# b = np.array([[5,2,2],[0,3,4]])
# print(a-b)
#
# # print(np.random.rand(10,21))
# print(np.linspace(-1, 1, 21))

# plt.plot([0.5,-0.5],[0.29,0.29])
# plt.plot([0.0,-0.5],[-0.58,0.29])
# plt.plot([0,0.5],[-0.58,0.29])
# # b
# plt.plot([0,-0.05],[0.058,-0.029])
# plt.plot([0,0.05],[0.058,-0.029])
# plt.plot([-0.05,0.05],[-0.029,-0.029])
# print(np.linalg.norm([0.05,0.087]))
# print(np.linalg.norm([0.5,0.87]))
# # plt.plot([-0.5,0.],[0.29,-0.58])
# # plt.plot([0.5,0.],[0.29,-0.58])
# # plt.plot([0.5,-0.5],[0.29,0.29])
#
# plt.show()

# x_uav = np.random.random((5, 8)) * 6 - 3  # 4 uav, includes x,y
# x_ugv = np.array([2.5, 1.45, 0, -2.5, 1.45, 0, 0, -2.9, 0])  # 9
# for i in range(5):
#     x_uav_temp = np.insert(x_uav[i], [2, 4, 6, 8], [7.5, 7.5, 7.5, 7.5])
#     print('x_ugv',x_uav[i])
#     print('x_ugv_temp', x_uav_temp)
#     cable_other_side = np.concatenate((x_uav_temp, x_ugv))
#     cable_other_side = np.reshape(cable_other_side, newshape=(7, 3))
#     print(cable_other_side)
#     print(np.shape(cable_other_side))

# a = np.arange(5)
# print(a)
# # [0 1 2 3]
#
# print(np.insert(a, 2, 10))
# # [  0   1 100   2   3]
#
# print(np.insert(a, 1, [10, 20, 30]))
# # [  0 100 101 102   1   2   3]
#
# print(np.insert(a, [0, 2, 4], [100, 101, 102]))
# # [100   0   1 101   2   3 102]


# input = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# print(np.shape(input))
# new_input = np.delete(input, 2, axis=1)
# print(new_input)
# print(input.flatten())


# list = [1,2,3]
# a,b,c = list
# print(a,b,c)

# input_ = np.array([[1,2,3],[3,4,5]])
# new_input = np.delete(input_, -1, axis=1)
# print(new_input)
# new_input = new_input.flatten()
# print(new_input)

# l = [np.zeros(4)]
# for i in range(12):
#     list = [[i, i+1, i+2, 5]]
#     list = np.concatenate((l, list), axis=0)
#     l = list
# print(list)

# list = np.array([1, 9, 1, 2, 3, 5, 9, 6])
# print(np.shape(list))
# list = np.reshape(list, (len(list), 1))
# print(list)
# appendex = []
# for i, l in enumerate(list):
#     if l > 5:
#         appendex.append(i)
# list = np.delete(list, appendex, axis=0)
# print(list)

# train_data = sio.loadmat(
#     '/home/firefly/chh_ws/src/plan_cdpr/scripts/data/static_02_value_input_2021-11-10-21:28:41.mat')
# # dict_keys(['__header__', '__version__', '__globals__', 'input_platform', 'input_pos', 'output_value', 'r_t', 'r_r'])            # print(train_data.keys())
# print(train_data.keys())
# r_r = train_data['r_r']
# r_t = train_data['r_t']
# print(np.shape(r_r))
# print(np.shape(r_t))
# for i in range(len(r_r[0])-1):
#     print('r_r', r_r[0, i])
#     print('r_t', r_t[0, i])
    # r_r[0, i] = (r_r[0, i] + r_r[0, i+1])/4
    # r_t[0, i] = (r_t[0, i] + r_t[0, i+1])/4

# plt.figure()
# plt.plot(r_r[0])
# plt.plot(r_t[0])

# plot_data = sio.loadmat(
#     '/home/firefly/chh_ws/src/plan_cdpr/scripts/data/10hz_network_trajectry_02_2021-11-11-12:11:16.mat')

# print(plot_data.keys())
# platform_pos = plot_data['platform_pos']
# steps = plot_data['steps']
# steps = [i for i in range(len(platform_pos))]
# print(np.shape(platform_pos))
# plot_pos = np.array(platform_pos).T

# plt.figure()
# plt.subplot(311)
# plt.plot(steps, plot_pos[0])
# plt.legend(labels=['x'], loc='best')
# plt.subplot(312)
# plt.plot(steps, plot_pos[1])
# plt.legend(labels=['y'], loc='best')
# plt.subplot(313)
# plt.plot(steps, plot_pos[2])
# plt.legend(labels=['z'], loc='best')
# plt.suptitle('platform_pos')
# plt.show()

# a = np.sum([[[-0.1]], [[-0.2]], [[-0.50186567]]])
# print(a)



