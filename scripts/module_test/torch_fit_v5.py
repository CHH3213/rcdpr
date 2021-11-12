#!/usr/bin/python3
# -*-coding:utf-8-*-
# @Project: plan_cdpr
# @File: torch_fit_v5.py---输入降维度，24维，小车z位置无意义，平台使用质心位置+姿态
# @Author: CHH
# @Time: 2021/11/4 下午16:19
'''
    训练完了，发现隐层越大，拟合的速度越是快，拟合的效果越是好
'''
from __future__ import print_function, absolute_import, division
from torch.autograd import Variable
import random
import sys
import matplotlib.pyplot as plt
import tf

sys.path.append('..')
sys.path.append('/home/chh3213/ros_wc/src/plan_cdpr/scripts')
from value_function import Value_function

sys.path.append('../scripts/data')
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from numpy.core.fromnumeric import shape
import torch
import scipy.io as sio
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import time
from distance_between_lines import *
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


# device = "cpu"
class Net(nn.Module):
    def __init__(self, n_input=42, n_hidden1=32, n_hidden2=32, n_output=1):
        super(Net, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_input, n_hidden1),
            nn.ReLU(),
            # nn.Linear(n_hidden1, n_hidden2),
            # nn.ReLU(),
            nn.Linear(n_hidden1, n_output),
        )

    def forward(self, input):
        return self.nn(input)

def env_generate():
    '''数据准备'''
    FileName = '/home/chh3213/ros_wc/src/plan_cdpr/scripts/data/value_data.mat'
    train_data = sio.loadmat(FileName)
    # dict_keys(['__header__', '__version__', '__globals__', 'input_platform', 'input_pos', 'output_value'])
    # print(train_data.keys())
    # input_pos：uav1~4, ugv1~3  input_platform：platform_point
    # print(np.shape(train_data['input_pos']))
    # print(train_data['input_pos'][0])
    x = []
    for i in range(500, 1500):
        T1 = np.array(train_data['input_pos']
                      [i]).flatten()  # 注意数据是否已经展平过
        T2 = np.array(train_data['input_platform'][i]).flatten()
        T_platform = np.concatenate((T1, T2))  # 前21为是位置变量，后21位是平台固定参数
        x.append(T_platform)
        # print(np.shape(T_platform))
    train_x = np.array(x)
    print(np.shape(x))
    # print(x[:, 0])
    # print(type(x))  # uav1~4, ugv1~3
    # print(np.shape(x))
    train_data['output_value'] = np.squeeze(train_data['output_value'])
    # print(np.shape(train_data['output_value']))
    train_y = np.array(train_data['output_value'][500:1500])
    train_y = np.reshape(train_y, (len(train_y), 1))
    print(np.shape(train_x))
    print(np.shape(train_y))
    return train_x,train_y

def random_generate(number, is_test=False):
    # np.random.seed(1)
    cable_length = np.array([5 for _ in range(7)])  # FIXME: 改为5， 原来是4.5
    train_x = []
    train_y = []
    x_uav = np.random.random((number, 12)) * 6 - 3  # 4 uav, includes x,y,z
    x_ugv = np.random.random((number, 6)) * 6 - 3  # 3ugv, not include z
    param_var = np.random.random((number, 6)) * 6 - 3  # platform pose x,y,z,r,p,y

    # 末端执行器固定点在其自身坐标系中的位置uav1~4,ugv1~3
    point_end_effector = np.array(
        [np.array([0.0, 0.29, 0.25]), np.array([-0.25, -0.145, 0.25]),
         np.array([0.25, -0.145, 0.25]), np.array([0.0, 0.0, 0.25]),
         np.array([0.0, 0.29, -0.25]), np.array([-0.25, -0.145, -0.25]),
         np.array([0.25, -0.145, -0.25])])
    if not is_test:
        if os.path.exists('/home/chh3213/ros_wc/src/plan_cdpr/scripts/data/fit_version5_v2_data.mat'):
            train_data = sio.loadmat('/home/chh3213/ros_wc/src/plan_cdpr/scripts/data/fit_version5_v2_data.mat')
            train_y = train_data['train_y']
            train_x = train_data['train_x']
        else:
            for i in range(number):
                x_ugv_temp = np.insert(x_ugv[i], [2, 4, 6], [0, 0, 0])  # 补足ugv位置信息
                # print(x_ugv[i])
                cable_other_side = np.concatenate((x_uav[i], x_ugv_temp))
                cable_other_side = np.reshape(cable_other_side, newshape=(7, 3))  # agent pos
                rot_center = np.array(param_var[i, 0:3])  # 末端执行器坐标系相对于基坐标系的位置
                pose_0 = tf.transformations.quaternion_from_euler(*param_var[i, 3:6])  # 四元数
                cable_one_side = np.empty((7, 3))  # 7个固定点，每个点是 3惟的
                for j in range(0, 7, 1):
                    vector = np.hstack([point_end_effector[j], 0])  # 将B点转换为四元数
                    # 通过四元数的旋转变换求得旋转后B’的四元数位置（相对于末端执行器坐标系）
                    rotated_vector = tf.transformations.quaternion_multiply(pose_0,
                                                                            tf.transformations.quaternion_multiply(
                                                                                vector,
                                                                                tf.transformations.quaternion_inverse(pose_0)))
                    # 将B’的四元数位置变成3维位置并进行坐标补偿将它变成相对于基坐标系的位置
                    cable_one_side[j] = np.delete(rotated_vector, 3) + rot_center

                # v1 = Value.cost_feasible_points(cable_one_side, cable_other_side, cable_length)
                v2 = Value.cost_cable_interference(cable_one_side, cable_other_side)
                # v3 = Value.cost_cable_length(cable_one_side, cable_other_side)
                # r1 = Value.r_t_AW(cable_one_side, cable_other_side)
                # r2 = Value.r_r_AW(cable_one_side, cable_other_side, rotation_center)
                # v4 = r1+r2
                y_value = v2
                x_value = np.concatenate((x_uav[i], x_ugv[i], param_var[i]))  #24dim
                train_x.append(x_value)
                train_y.append(y_value)
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            train_y = train_y.reshape((number, 1))
            print(np.shape(train_y))
            print(np.shape(train_x))
            sio.savemat('/home/chh3213/ros_wc/src/plan_cdpr/scripts/data' + '/fit_version5_v2_data.mat',
                        {'train_x': train_x, 'train_y': train_y})
        print('处理完毕')
        print('train_x shape ', np.shape(train_x))
        print('train_y shape', np.shape(train_y))
    else:
        for i in range(number):
            x_ugv_temp = np.insert(x_ugv[i], [2, 4, 6], [0, 0, 0])  # 补足ugv位置信息
            # print(x_ugv[i])
            cable_other_side = np.concatenate((x_uav[i], x_ugv_temp))
            cable_other_side = np.reshape(cable_other_side, newshape=(7, 3))  # agent pos
            rot_center = np.array(param_var[i, 0:3])  # 末端执行器坐标系相对于基坐标系的位置
            pose_0 = tf.transformations.quaternion_from_euler(*param_var[i, 3:6])  # 四元数
            cable_one_side = np.empty((7, 3))  # 7个固定点，每个点是 3惟的
            for j in range(0, 7, 1):
                vector = np.hstack([point_end_effector[j], 0])  # 将B点转换为四元数
                # 通过四元数的旋转变换求得旋转后B’的四元数位置（相对于末端执行器坐标系）
                rotated_vector = tf.transformations.quaternion_multiply(pose_0,
                                                                        tf.transformations.quaternion_multiply(
                                                                            vector,
                                                                            tf.transformations.quaternion_inverse(
                                                                                pose_0)))
                # 将B’的四元数位置变成3维位置并进行坐标补偿将它变成相对于基坐标系的位置
                cable_one_side[j] = np.delete(rotated_vector, 3) + rot_center

            # v1 = Value.cost_feasible_points(cable_one_side, cable_other_side, cable_length)
            v2 = Value.cost_cable_interference(cable_one_side, cable_other_side)
            # v3 = Value.cost_cable_length(cable_one_side, cable_other_side)
            # r1 = Value.r_t_AW(cable_one_side, cable_other_side)
            # r2 = Value.r_r_AW(cable_one_side, cable_other_side, rotation_center)
            # v4 = r1+r2
            y_value = v2
            x_value = np.concatenate((x_uav[i], x_ugv[i], param_var[i]))  # 24dim
            train_x.append(x_value)
            train_y.append(y_value)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_y = train_y.reshape((number, 1))
        print(np.shape(train_y))
        print(np.shape(train_x))

    return train_x, train_y


if __name__ == '__main__':
    Value = Value_function()
    random_data = True
    train = True
    train = False
    if train == True:
        epoches = 50000
        if random_data == False:
            train_x, train_y = env_generate()
        else:
            '''################随机生成数据训练#####################'''
            number = 10000
            train_x, train_y = random_generate(number, False)

        train_x = torch.tensor(train_x, device=device, dtype=torch.float32)
        train_y = torch.tensor(train_y, device=device, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(train_x, train_y)
        loader = torch.utils.data.DataLoader(ds, batch_size=number, shuffle=True)
        '''++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
        '''开始train 部分'''
        net = Net(24, 512, 1).to(device)
        # optimizer 优化
        if os.path.exists('/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/model_24dim/torch_fit_value2.pt'):
            print('load model!!!')
            print('**********************')
            net = torch.load('/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/model_24dim/torch_fit_value2.pt')
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, weight_decay=0.001)
        # 每跑50000个step就把学习率乘以0.9
        # scheduler = ReduceLROnPlateau(optimizer, 'min',patience=100,factor=0.5)
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)
        # loss funaction
        loss_funaction = torch.nn.MSELoss()
        step = 0
        # 尝试
        for i in range(epoches):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = Variable(data), Variable(target)
                pridect_y = net(data)  # 喂入训练数据 得到预测的y值
                optimizer.zero_grad()  # 为下一次训练清除上一步残余更新参数
                loss = loss_funaction(pridect_y, target)  # 计算损失
                loss.backward()  # 误差反向传播，计算梯度
                optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
                # if i <= 5000:
                #     scheduler.step()
                # scheduler.step()
                # if (batch_idx+1) % 100 == 0:
                #     print(batch_idx, loss.item())
            # pridect_y = net(Variable(train_x,requires_grad = False))  # 喂入训练数据 得到预测的y值 FIXME: 用来干啥？

            if i % 50 == 0:
                # plt.cla()
                # plt.plot(train_y.cpu().numpy())
                # plt.plot(pridect_y.data.cpu().numpy(), 'r-')
                # plt.pause(0.3)
                # print("已训练{}步 | loss：{} | y_data:{} | predict_y:{}.".format(i, loss, y_data.item().sum(), pridect_y.item().sum()))
                print("已训练{}步 | loss：{} .".format(i, loss))
                torch.save(net, '/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/model_24dim/torch_fit_value2.pt')
        '''++++++++++++++++++++++++++++++'''

        torch.save(net, '/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/model_24dim/torch_fit_value2.pt')
    else:
        '''#####################随机生成测试数据########################'''
        net = Net(21, 512, 1).to(device)
        net = torch.load('/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/model_24dim/torch_fit_value2.pt')
        plot_y = []
        plot_y_ = []
        plot_loss = []
        rotation_center = np.array([0, 0, 3])  # FIXME: 增加， r_r_AW 需要，但原来的没有
        number = 128
        test_x,test_y = random_generate(number,True)
        for i in range(number):

            '''2环境获取的数据'''
            x_data = torch.tensor(test_x[i], device=device,
                                  dtype=torch.float32, requires_grad=True)
            y_data = torch.tensor(test_y[i], device=device, dtype=torch.float32)
            y_ = net(x_data)
            y_.backward()  # 反向传播计算梯度
            # x_data.grad.zero_()
            grads_1 = x_data.grad
            print('+++++++++++++++')
            print('target', test_y[i])
            print('approximate', y_.item())
            plot_y.append(test_y[i])
            plot_y_.append(y_.item())
            # 梯度
            # print('torch_grads', grads_1)
            # print('func_grads', grads_2)
            # 误差
            print('func_mse: {}'.format(F.mse_loss(y_data, y_)))
            plot_loss.append(F.mse_loss(y_data, y_).item())
            print('===================')
        plt.title('predict')
        plt.plot(plot_y, 'b-')
        plt.plot(plot_y_, 'r-')
        plt.legend(['real_y', 'predict_y'])
        plt.figure()
        plt.plot(plot_loss)
        plt.legend(['loss'])
        plt.title('loss')
        plt.show()
