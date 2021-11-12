#!/usr/bin/python3
# -*-coding:utf-8-*-
# @Project: ugv_uav
# @File: torch_fit.py---使用torch 神经网络对函数拟合求梯度
# @Author: CHH
# @Time: 2021/10/25 上午10:57
'''
    训练完了，发现隐层越大，拟合的速度越是快，拟合的效果越是好
'''
from __future__ import print_function, absolute_import, division
from torch.autograd import Variable
import random
import os
import sys
import matplotlib.pyplot as plt

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
from torch_data import train_data

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


def calcus_inter(net, cables_one_side, cables_other_side):
    lines = np.array([np.array([cables_one_side[i], cables_other_side[i]])
                      for i in range(len(cables_one_side))])
    lines = np.concatenate()
    sum_list = []
    for i in range(len(lines)):
        assert len(lines[i]) % 2 == 0, "线段是由两个端点描述，请检查端点数是否正确！"
        for j in range(i + 1, len(lines)):
            _, _, d_ij = net(
                lines[i, 0], lines[i, 1], lines[j, 0], lines[j, 1], )
            f_ij = np.exp(-(d_ij - 0.1) / 0.017)
            sum_list.append(f_ij)
    # print('sum_list:{}'.format(sum_list))
    sum_f = np.sum(sum_list)
    return sum_f


if __name__ == '__main__':
    Value = Value_function()
    random_data = True
    train = True
    # train = False
    if train == True:
        epoches = 50000
        train_x, train_y, number = train_data(is_random=random_data, is_random_load=True, fit_v_all=False,
                                              fit_v1=False, fit_v2=True, fit_v3=False,
                                              fit_v4=False)

        train_x = torch.tensor(train_x, device=device, dtype=torch.float32)
        train_y = torch.tensor(train_y, device=device, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(train_x, train_y)
        loader = torch.utils.data.DataLoader(ds, batch_size=number, shuffle=True)
        '''++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
        '''开始train 部分'''
        net = Net(42, 512, 1).to(device)
        # optimizer 优化
        # if os.path.exists('/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/env_torch_fit_value2.pt'):
        #     print('load model!!!')
        #     print('**********************')
        #     net = torch.load('/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/env_torch_fit_value2.pt')
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
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

            if i % 100 == 0:
                # plt.cla()
                # plt.plot(train_y.cpu().numpy())
                # plt.plot(pridect_y.data.cpu().numpy(), 'r-')
                # plt.pause(0.3)
                # print("已训练{}步 | loss：{} | y_data:{} | predict_y:{}.".format(i, loss, y_data.item().sum(), pridect_y.item().sum()))
                print("已训练{}步 | loss：{} .".format(i, loss))
                torch.save(net, '/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/env_torch_fit_value2.pt')
        '''++++++++++++++++++++++++++++++'''

        torch.save(net, '/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/env_torch_fit_value2.pt')
    else:
        # test
        # FileName = '../scripts/data/value_data.mat'
        # test_data = sio.loadmat(FileName)
        # # dict_keys(['__header__', '__version__', '__globals__', 'input_platform', 'input_pos', 'output_value'])
        # # print(test_data.keys())
        # # input_pos：uav1~4, ugv1~3  input_platform：platform_point
        # # print(np.shape(test_data['input_pos']))
        # # print(test_data['input_pos'][0])
        # x = []
        # for i in range(500, 1500):
        #     T1 = np.array(test_data['input_pos'][i]).flatten()  # 注意数据是否已经展平过
        #     T2 = np.array(test_data['input_platform'][i]).flatten()
        #     T_platform = np.concatenate((T1, T2))  # 前21为是位置变量，后21位是平台固定参数
        #     x.append(T_platform)
        #     # print(np.shape(T_platform))
        # test_x = np.array(x)
        # # print(np.shape(x))
        # # print(x[:, 0])
        # print(type(x))  # uav1~4, ugv1~3
        # # print(np.shape(x))
        # test_data['output_value'] = np.squeeze(test_data['output_value'])
        # # print(np.shape(train_data['output_value']))
        # test_y = np.array(test_data['output_value'][500:1500])
        '''#####################随机生成测试数据########################'''
        x = np.random.random((128, 21)) * 6 - 3
        param_var = np.random.random((128, 21)) * 6 - 3

        cable_length = np.array([5 for _ in range(7)])
        net = Net(42, 512, 1).to(device)
        net = torch.load('/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/torch_fit_v3.pt')
        plot_y = []
        plot_y_ = []
        plot_loss = []
        rotation_center = np.array([0, 0, 3])  # FIXME: 增加， r_r_AW 需要，但原来的没有
        for i in range(len(x)):
            cable_other_side = np.reshape(x[i], newshape=(7, 3))
            cable_one_side = np.reshape(param_var[i], newshape=(7, 3))
            # a = np.reshape(x[i], newshape=(2, 3))
            # b = np.reshape(param_var[i], newshape=(2, 3))
            # print(np.shape(cable_other_side))
            # v1 = Value.cost_feasible_points(cable_one_side, cable_other_side, cable_length)
            v2 = Value.cost_cable_interference(cable_one_side, cable_other_side)
            # v3 = Value.cost_cable_length(cable_one_side, cable_other_side)
            # r1 = Value.r_t_AW(cable_one_side, cable_other_side)
            # r2 = Value.r_r_AW(cable_one_side, cable_other_side, rotation_center)
            # _, _, d_ij = closestDistanceBetweenLines(
            #     a[0], a[1], b[0], b[1], clampAll=True)
            test_y = v2
            test_x = np.concatenate((x[i], param_var[i]))
            # test_x = x[i]
            '''2环境获取的数据'''
            x_data = torch.tensor(test_x, device=device,
                                  dtype=torch.float32, requires_grad=True)
            y_data = torch.tensor(test_y, device=device, dtype=torch.float32)
            y_ = net(x_data)
            y_.backward()  # 反向传播计算梯度
            # x_data.grad.zero_()
            grads_1 = x_data.grad
            print('+++++++++++++++')
            print('target', test_y)
            print('approximate', y_.item())
            plot_y.append(test_y)
            plot_y_.append(y_.item())
            # 梯度
            print('torch_grads', grads_1)
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
