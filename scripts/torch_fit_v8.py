#!/usr/bin/python3
# -*-coding:utf-8-*-
# @Project: plan_cdpr
# @File: torch_fit_v6.py---输入降维度，14维，不考虑小车，小车固定，平台使用质心位置+姿态，无人机不关系高度
# @Author: CHH
# @Time: 2021/11/4 下午21：39
'''
    训练完了，发现隐层越大，拟合的速度越是快，拟合的效果越是好
'''

import matplotlib.pyplot as plt
import tf
from value_function import Value_function
import numpy as np

import scipy.io as sio
import time
from distance_between_lines import *
import copy
import random


Value = Value_function()

def generate_x_circle0(number):
    rad = 3.0
    x_uav_1_t = np.random.random((number))*np.pi*2  
    x_uav_1_d = np.random.random((number))*rad 
    x_uav_1_x = np.cos(x_uav_1_t) * x_uav_1_d + (-2.5)
    x_uav_1_y = np.sin(x_uav_1_t) * x_uav_1_d + (2.5)

    x_uav_2_t = np.random.random((number))*np.pi*2 
    x_uav_2_d = np.random.random((number))*rad
    x_uav_2_x = np.cos(x_uav_2_t) * x_uav_2_d + (-2.5)
    x_uav_2_y = np.sin(x_uav_2_t) * x_uav_2_d + (-2.5)

    x_uav_3_t = np.random.random((number))*np.pi*2
    x_uav_3_d = np.random.random((number))*rad
    x_uav_3_x = np.cos(x_uav_3_t) * x_uav_3_d + (2.5)
    x_uav_3_y = np.sin(x_uav_3_t) * x_uav_3_d + (-2.5)

    x_uav_4_t = np.random.random((number))*np.pi*2
    x_uav_4_d = np.random.random((number))*rad
    x_uav_4_x = np.cos(x_uav_4_t) * x_uav_4_d + (2.5)
    x_uav_4_y = np.sin(x_uav_4_t) * x_uav_4_d + (2.5)

    param_var_x = 0 + np.random.random((number))*6.0-3.0
    param_var_y = 0 + np.random.random((number))*6.0-3.0
    param_var_z = 4 + np.random.random((number))*1-0.5

    x_uav_temp = [x_uav_1_x, x_uav_1_y, x_uav_2_x,
                  x_uav_2_y, x_uav_3_x, x_uav_3_y, x_uav_4_x, x_uav_4_y]
    x_uav = np.array(x_uav_temp).T

    param_var_temp = [param_var_x, param_var_y, param_var_z]
    param_var = np.array(param_var_temp).T

    print(np.shape(x_uav))
    print(np.shape(param_var))

    return x_uav, param_var


def generate_x_circle1(number):
    rad = 0.45
    x_uav_1_t = np.random.random((number))*np.pi*2
    x_uav_1_d = np.random.random((number))*rad+2.2
    x_uav_1_x = np.cos(x_uav_1_t) * x_uav_1_d + (-2.5)
    x_uav_1_y = np.sin(x_uav_1_t) * x_uav_1_d + (2.5)

    x_uav_2_t = np.random.random((number))*np.pi*2
    x_uav_2_d = np.random.random((number))*rad+2.2
    x_uav_2_x = np.cos(x_uav_2_t) * x_uav_2_d + (-2.5)
    x_uav_2_y = np.sin(x_uav_2_t) * x_uav_2_d + (-2.5)

    x_uav_3_t = np.random.random((number))*np.pi*2
    x_uav_3_d = np.random.random((number))*rad+2.2
    x_uav_3_x = np.cos(x_uav_3_t) * x_uav_3_d + (2.5)
    x_uav_3_y = np.sin(x_uav_3_t) * x_uav_3_d + (-2.5)

    x_uav_4_t = np.random.random((number))*np.pi*2
    x_uav_4_d = np.random.random((number))*rad+2.2
    x_uav_4_x = np.cos(x_uav_4_t) * x_uav_4_d + (2.5)
    x_uav_4_y = np.sin(x_uav_4_t) * x_uav_4_d + (2.5)

    param_var_x = 0 + np.random.random((number))*6.0-3.0
    param_var_y = 0 + np.random.random((number))*6.0-3.0
    param_var_z = 4 + np.random.random((number))*1-0.5

    x_uav_temp = [x_uav_1_x, x_uav_1_y, x_uav_2_x,
                  x_uav_2_y, x_uav_3_x, x_uav_3_y, x_uav_4_x, x_uav_4_y]
    x_uav = np.array(x_uav_temp).T

    param_var_temp = [param_var_x, param_var_y, param_var_z]
    param_var = np.array(param_var_temp).T

    print(np.shape(x_uav))
    print(np.shape(param_var))

    return x_uav, param_var


def generate_x0(number):
    
    # all random
    # x_uav = np.random.random((number, 8)) * 10- 5  # 4 uav, includes x,y
    # generate random points around the trajectory they walking along
    # Case 1: random step
    x_uav_1_x = -2.5 + np.random.random((number))*3.54-1.77  # -4.25, -0.25
    x_uav_2_x = -2.5 + np.random.random((number))*3.54-1.77   # -1.5, 2.5
    x_uav_3_x = 2.5 + np.random.random((number))*3.54-1.77   # 1, 5
    x_uav_4_x = 2.5 + np.random.random((number))*3.54-1.77   # -2, 2
    x_uav_1_y = 2.5 + np.random.random((number))*3.54-1.77   # -0.3, 3.7
    x_uav_2_y = -2.5 + np.random.random((number))*3.54-1.77   # -4.7, -0.7
    x_uav_3_y = -2.5 + np.random.random((number))*3.54-1.77   # -0.3, 3.7
    x_uav_4_y = 2.5 + np.random.random((number))*3.54-1.77   # -1.8, 2.2


    # Case 2: step = 0.001
    # x_uav_1_x = np.arange(start=-3.5, stop=-0.5, step=0.001)
    # x_uav_2_x = np.arange(start=-1.5, stop=1.5, step=0.001)
    # x_uav_3_x = np.arange(start=0.5, stop=3.5, step=0.001)
    # x_uav_4_x = np.arange(start=-1.5, stop=1.5, step=0.001)
    # x_uav_1_y = np.arange(start=0., stop=3., step=0.001)
    # x_uav_2_y = np.arange(start=-4., stop=-1., step=0.001)
    # x_uav_3_y = np.arange(start=0., stop=3., step=0.001)
    # x_uav_4_y = np.arange(start=-1., stop=2., step=0.001)
    # for i in range(5):
    #     x_uav_1_x = np.concatenate((x_uav_1_x, x_uav_1_x))
    #     x_uav_1_y = np.concatenate((x_uav_1_y, x_uav_1_y))
    #     x_uav_2_x = np.concatenate((x_uav_2_x, x_uav_2_x))
    #     x_uav_2_y = np.concatenate((x_uav_2_y, x_uav_2_y))
    #     x_uav_3_x = np.concatenate((x_uav_3_x, x_uav_3_x))
    #     x_uav_3_y = np.concatenate((x_uav_3_y, x_uav_3_y))
    #     x_uav_4_x = np.concatenate((x_uav_4_x, x_uav_4_x))
    #     x_uav_4_y = np.concatenate((x_uav_4_y, x_uav_4_y))
    x_uav_temp = [x_uav_1_x, x_uav_1_y, x_uav_2_x,
                  x_uav_2_y, x_uav_3_x, x_uav_3_y, x_uav_4_x, x_uav_4_y]
    x_uav = np.array(x_uav_temp).T
    # x_uav_ = []
    # for i in range(len(x_uav_temp)):
    #     uav_temp = x_uav_temp[i]
    #     # print(len(uav_temp))
    #     # np.random.shuffle(uav_temp)
    #     x_uav_.append(copy.deepcopy(uav_temp))
    # # print(x_uav_)
    # x_uav = np.array(x_uav_).T

    

    #all random
    # param_var = np.random.random((number, 6)) * 2-1 # platform pose x,y,z,r,p,y
    # generate random points around the trajectory they walking along
    # Case 1: random step
    param_var_x = 0 + np.random.random((number))*6.0-3.0
    param_var_y = 0 + np.random.random((number))*6.0-3.0
    param_var_z = 4 + np.random.random((number))*1-0.5

    # Case 2: step = 0.001
    # param_var_x = np.random.random((96000))*2-1
    # param_var_y = np.random.random((96000))*2-1
    # param_var_z = np.random.random((96000))*2-1+4
    # param_var_euler = np.random.random((96000))*2-1
    ### arange
    # param_var_x = np.arange(start=-0.1, stop=0.5, step=0.001)
    # param_var_y = np.arange(start=-0.3, stop=0.3, step=0.001)
    # param_var_z = np.arange(start=3.5, stop=4.5, step=0.001)
    # param_var_euler = np.arange(start=-0.5, stop=0.5, step=0.001)
    # for i in range(8):
    #     param_var_x = np.concatenate((param_var_x, param_var_x))
    #     param_var_y = np.concatenate((param_var_y, param_var_y))
    #     param_var_z = np.concatenate((param_var_z, param_var_z))
    #     param_var_euler = np.concatenate((param_var_euler, param_var_euler))
    # param_var_temp = [param_var_x,param_var_y,param_var_z,param_var_euler,copy.deepcopy(param_var_euler),copy.deepcopy(param_var_euler)]
    param_var_temp = [param_var_x, param_var_y, param_var_z]
    param_var = np.array(param_var_temp).T
    # param_var_ = []
    # for i in range(6):
    #     temp = param_var_temp[i]
    #     # print('r',len(temp))
    #     # temp = temp[:96000]
    #     np.random.shuffle(temp)
    #     param_var_.append(copy.deepcopy(temp))
    # param_var = np.array(param_var_).T

    print(np.shape(x_uav))
    print(np.shape(param_var))

    return x_uav, param_var


def generate_x1(number):
    num = int(number/2)
    # all random
    # x_uav = np.random.random((number, 8)) * 10- 5  # 4 uav, includes x,y
    # generate random points around the trajectory they walking along
    # Case 1: random step
    x_uav_1_x0 = -2.5 + np.random.random((num))*0.83 + 1.77 # -4.25, -0.25
    x_uav_1_x1 = -2.5 + np.random.random((num))*(-0.83) - 1.77 # -4.25, -0.25
    # print(np.shape(x_uav_1_x1))
    x_uav_1_x = np.concatenate((x_uav_1_x0, x_uav_1_x1))
    # print(np.shape(x_uav_1_x))

    x_uav_2_x0 = -2.5 + np.random.random((num))*0.83 + 1.77 # -1.5, 2.5
    x_uav_2_x1 = -2.5 + np.random.random((num))*(-0.83) - 1.77 # -1.5, 2.5
    x_uav_2_x = np.concatenate((x_uav_2_x0, x_uav_2_x1))

    x_uav_3_x0 = 2.5 + np.random.random((num))*0.83 + 1.77 # 1, 5
    x_uav_3_x1 = 2.5 + np.random.random((num))*(-0.83) - 1.77 # 1, 5
    x_uav_3_x = np.concatenate((x_uav_3_x0, x_uav_3_x1))

    x_uav_4_x0 = 2.5 + np.random.random((num))*0.83 + 1.77 # -2, 2
    x_uav_4_x1 = 2.5 + np.random.random((num))*(-0.83) - 1.77  # -2, 2
    x_uav_4_x = np.concatenate((x_uav_4_x0, x_uav_4_x1))

    x_uav_1_y0 = 2.5 + np.random.random((num))*0.83 + 1.77 # -0.3, 3.7
    x_uav_1_y1 = 2.5 + np.random.random((num))*(-0.83) - 1.77 # -0.3, 3.7
    x_uav_1_y = np.concatenate((x_uav_1_y0, x_uav_1_y1))

    x_uav_2_y0 = -2.5 + np.random.random((num))*0.83 + 1.77 # -4.7, -0.7
    x_uav_2_y1 = -2.5 + np.random.random((num))*(-0.83) - 1.77 # -4.7, -0.7
    x_uav_2_y = np.concatenate((x_uav_2_y0, x_uav_2_y1))

    x_uav_3_y0 = -2.5 + np.random.random((num))*0.83 + 1.77 # -0.3, 3.7
    x_uav_3_y1 = -2.5 + np.random.random((num))*(-0.83) - 1.77 # -0.3, 3.7
    x_uav_3_y = np.concatenate((x_uav_3_y0, x_uav_3_y1))

    x_uav_4_y0 = 2.5 + np.random.random((num))*0.83 + 1.77 # -1.8, 2.2
    x_uav_4_y1 = 2.5 + np.random.random((num))*(-0.83) - 1.77 # -1.8, 2.2
    x_uav_4_y = np.concatenate((x_uav_4_y0, x_uav_4_y1))

    # Case 2: step = 0.001
    # x_uav_1_x = np.arange(start=-3.5, stop=-0.5, step=0.001)
    # x_uav_2_x = np.arange(start=-1.5, stop=1.5, step=0.001)
    # x_uav_3_x = np.arange(start=0.5, stop=3.5, step=0.001)
    # x_uav_4_x = np.arange(start=-1.5, stop=1.5, step=0.001)
    # x_uav_1_y = np.arange(start=0., stop=3., step=0.001)
    # x_uav_2_y = np.arange(start=-4., stop=-1., step=0.001)
    # x_uav_3_y = np.arange(start=0., stop=3., step=0.001)
    # x_uav_4_y = np.arange(start=-1., stop=2., step=0.001)
    # for i in range(5):
    #     x_uav_1_x = np.concatenate((x_uav_1_x, x_uav_1_x))
    #     x_uav_1_y = np.concatenate((x_uav_1_y, x_uav_1_y))
    #     x_uav_2_x = np.concatenate((x_uav_2_x, x_uav_2_x))
    #     x_uav_2_y = np.concatenate((x_uav_2_y, x_uav_2_y))
    #     x_uav_3_x = np.concatenate((x_uav_3_x, x_uav_3_x))
    #     x_uav_3_y = np.concatenate((x_uav_3_y, x_uav_3_y))
    #     x_uav_4_x = np.concatenate((x_uav_4_x, x_uav_4_x))
    #     x_uav_4_y = np.concatenate((x_uav_4_y, x_uav_4_y))
    x_uav_temp = [x_uav_1_x, x_uav_1_y, x_uav_2_x,
                  x_uav_2_y, x_uav_3_x, x_uav_3_y, x_uav_4_x, x_uav_4_y]
    x_uav = np.array(x_uav_temp).T
    # x_uav_ = []
    # for i in range(len(x_uav_temp)):
    #     uav_temp = x_uav_temp[i]
    #     # print(len(uav_temp))
    #     # np.random.shuffle(uav_temp)
    #     x_uav_.append(copy.deepcopy(uav_temp))
    # # print(x_uav_)
    # x_uav = np.array(x_uav_).T

    #all random
    # param_var = np.random.random((number, 6)) * 2-1 # platform pose x,y,z,r,p,y
    # generate random points around the trajectory they walking along
    # Case 1: random step
    param_var_x = 0 + np.random.random((number))*6.0-3.0
    param_var_y = 0 + np.random.random((number))*6.0-3.0
    param_var_z = 4 + np.random.random((number))*1-0.5

    # Case 2: step = 0.001
    # param_var_x = np.random.random((96000))*2-1
    # param_var_y = np.random.random((96000))*2-1
    # param_var_z = np.random.random((96000))*2-1+4
    # param_var_euler = np.random.random((96000))*2-1
    ### arange
    # param_var_x = np.arange(start=-0.1, stop=0.5, step=0.001)
    # param_var_y = np.arange(start=-0.3, stop=0.3, step=0.001)
    # param_var_z = np.arange(start=3.5, stop=4.5, step=0.001)
    # param_var_euler = np.arange(start=-0.5, stop=0.5, step=0.001)
    # for i in range(8):
    #     param_var_x = np.concatenate((param_var_x, param_var_x))
    #     param_var_y = np.concatenate((param_var_y, param_var_y))
    #     param_var_z = np.concatenate((param_var_z, param_var_z))
    #     param_var_euler = np.concatenate((param_var_euler, param_var_euler))
    # param_var_temp = [param_var_x,param_var_y,param_var_z,param_var_euler,copy.deepcopy(param_var_euler),copy.deepcopy(param_var_euler)]
    param_var_temp = [param_var_x, param_var_y, param_var_z]
    param_var = np.array(param_var_temp).T
    # param_var_ = []
    # for i in range(6):
    #     temp = param_var_temp[i]
    #     # print('r',len(temp))
    #     # temp = temp[:96000]
    #     np.random.shuffle(temp)
    #     param_var_.append(copy.deepcopy(temp))
    # param_var = np.array(param_var_).T

    print(np.shape(x_uav))
    print(np.shape(param_var))

    return x_uav, param_var


def random_generate(number, is_test=False):
    # np.random.seed(1)
    train_x = []
    train_y = []
    cable_length = np.array([5 for _ in range(7)]) 
    x_ugv = np.array([2.5, 1.45, 0.06, -2.5, 1.45, 0.06, 0, -2.9, 0.06])  # 9

    
    # x_uav = np.array([[-1.90000000e+00,  1.79641016e+00, 6.00000000e-01, - 2.55358984e+00,
    #                  3.10000000e+00, 1.79641016e+00, 6.00000000e-01, 3.46410162e-01]])
    # param_var = np.array([[0.29546405 ,0.18800748, 4.04525054, 0.20645114, -1.12417206, 2.32108617]])
    # v1,v2,r_t_AW 0.0 2.9640723125791224 -3.9169565066392935

    # 末端执行器固定点在其自身坐标系中的位置uav1~4,ugv1~3
    point_end_effector = np.array([[0, 0, 0] for _ in range(7)])
    temp_flag = 0
    if not is_test:
        if not True:
            pass
        else:
            start = time.time()
            while not (len(train_y) >= number):
                if (len(train_y) <= number):
                    x_uav, param_var = generate_x_circle0(1000)
                else:
                    temp_flag = 1
                    x_uav, param_var = generate_x_circle1(1000)

                print('len:', np.shape(x_uav)[0])

                for i in range(np.shape(x_uav)[0]):
                    if i % 10 == 5:
                        if len(train_y) == 0:
                            pass
                        else:
                            print('No. {}\{}\t y {}'.format(len(train_y),temp_flag, train_y[-1]))
                        # start = time.time()
                    x_uav_temp = np.insert(x_uav[i], [2, 4, 6, 8], [8.0, 8.0, 8.0, 8.0])  # 补足uav位置信息x,y,z
                    cable_other_side = np.concatenate((x_uav_temp, x_ugv))
                    cable_other_side = np.reshape(cable_other_side, newshape=(7, 3))  # agent pos
                    rot_center = np.array(param_var[i, 0:3])  # 末端执行器坐标系相对于基坐标系的位置
                    cable_one_side = np.array([rot_center for i in range(7)])
                    # pose_0 = tf.transformations.quaternion_from_euler(
                    #     *param_var[i, 3:6])  # 四元数
                    # cable_one_side = np.empty((7, 3))  # 7个固定点，每个点是 3惟的
                    
                    # for j in range(0, 7, 1):
                    #     vector = np.hstack([point_end_effector[j], 0])  # 将B点转换为四元数
                    #     # 通过四元数的旋转变换求得旋转后B’的四元数位置（相对于末端执行器坐标系）
                    #     rotated_vector = tf.transformations.quaternion_multiply(pose_0,
                    #                                                             tf.transformations.quaternion_multiply(
                    #                                                                 vector,
                    #                                                                 tf.transformations.quaternion_inverse(pose_0)))
                    #     # 将B’的四元数位置变成3维位置并进行坐标补偿将它变成相对于基坐标系的位置
                    #     cable_one_side[j] = np.delete(
                    #         rotated_vector, 3) + rot_center
                    pose_0 = np.array([0,0,0,1]) 
                    Value.set_jacobian_param(point_end_effector,pose_0,rot_center)
                    v2 = Value.cost_feasible_points(cable_other_side)
                    v1 = Value.cost_cable_interference(cable_one_side, cable_other_side)
                    r1 = Value.r_t_AW(cable_one_side, cable_other_side)
                    v4 = r1
                    y_value = v1+v2+v4
                    # if y_value>100:
                    #     # print('point_end_effector,pose_0,rot_center',point_end_effector,pose_0,rot_center)
                    #     print('cable_other_side', cable_other_side)
                    #     print('y_value:{}\t v1: {}\t v2: {}\t rt: {}'.format(y_value,v1,v2,v4))
                    if y_value< 600:
                        x_value = np.concatenate((x_uav[i], param_var[i]))  # 11dim
                        train_x.append(x_value)
                        train_y.append(y_value)
                train_xx = np.array(train_x)
                train_yy = np.array(train_y)
                train_yy = train_yy.reshape((np.shape(train_xx)[0], 1))
                print(np.shape(train_yy))
                print(np.shape(train_xx))
                print('save data')
                sio.savemat('data/random_train_data_18.mat',
                            {'train_x': train_xx, 'train_y': train_yy})
        print('处理完毕')
        print('train_x shape ', np.shape(train_x))
        print('train_y shape', np.shape(train_y))
    else:
        pass

        train_y = train_y.reshape(number, 1)
        print(np.shape(train_y))
        print(np.shape(train_x))

    return train_x, train_y


if __name__ == '__main__':
    # payload_gravity = 0.98
    # m = payload_gravity / 9.8
    # d_min = 5
    # mu_t, mu_r, mu_d, mu_a, mu_l = 1.0, 1.0, 0.02, 0.2, 0.2
    # t_min = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # t_max = np.array([5, 5, 5, 5, 5, 5, 5])
    # target_length_list = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])


    
    random_data = True
    train = True
    # train = False
    if train == True:
        epoches = 3600000
        if random_data == False:
            pass
        else:
            '''################随机生成数据训练#####################'''
            number = 10000
            train_x, train_y = random_generate(number, False)

    # x = np.array([[1,2,3],[4,5,6]])
    # y = np.array([[4, 5, 6], [3, 2, 1]])
    # print(x*np.cos(y))
