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
import sys
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
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
class Neuro_net(torch.nn.Module):
    """搭建神经网络"""
    def __init__(self, n_feature=42, n_hidden=32, n_output=1):
        super(Neuro_net, self).__init__()   # 继承__init__功能
        self.hidden_layer1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden_layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.output_layer = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = F.sigmoid(x)
        x = self.hidden_layer2(x)
        x = F.sigmoid(x)

        pridect_y = self.output_layer(x)
        return pridect_y

if __name__ == '__main__':
    Value = Value_function()
    random_data = False
    train = True
    if train ==True:
        epoches = 100
        if random_data ==False:
            '''数据准备'''
            FileName = '/home/chh3213/ros_wc/src/plan_cdpr/scripts/data/value_data.mat'
            train_data = sio.loadmat(FileName)
            # dict_keys(['__header__', '__version__', '__globals__', 'input_platform', 'input_pos', 'output_value'])
            # print(train_data.keys())
            # input_pos：uav1~4, ugv1~3  input_platform：platform_point
            # print(np.shape(train_data['input_pos']))
            # print(train_data['input_pos'][0])
            x = []
            for i in range(500, 1499):
                T1 = np.array(train_data['input_pos'][i]).flatten()  # 注意数据是否已经展平过
                T2 = np.array(train_data['input_platform'][i]).flatten()
                T_platform = np.concatenate((T1, T2))  # 前21为是位置变量，后21位是平台固定参数
                x.append(T_platform)
                # print(np.shape(T_platform))
            train_x = np.array(x)
            print(np.shape(x))
            # print(x[:, 0])
            print(type(x))  # uav1~4, ugv1~3
            # print(np.shape(x))
            train_data['output_value'] = np.squeeze(train_data['output_value'])
            # print(np.shape(train_data['output_value']))
            train_y = np.array(train_data['output_value'][500:1500])
        else:
            '''################随机生成数据训练#####################'''
            number = 5000
            np.random.seed(1)
            batchsize = 32
            #
            # x = np.random.random((number, batchsize, 21)) * 3
            # param_var = np.random.random((number, batchsize, 21)) * 3
            # # ############
            # # x = np.rand(number,21)
            # # param_var = np.rand(number,21)
            # # index = [i for i in range(x)]
            # # random.shuffle(index)
            # # x = x[index]
            # # param_var = param_var[index]
            # # #################
            # # print(x[0])
            # # print(param_var[0])
            # print(shape(x))
            # train_x = []
            # train_y = []
            # cable_length = np.array([4.5 for _ in range(7)])
            # for i in range(number):
            #     '''1随机的数据--需要做的处理'''
            #     x_list = []
            #     y_list = []
            #     for j in range(batchsize):
            #         cable_other_side = np.reshape(x[i][j], newshape=(7, 3))
            #         cable_one_side = np.reshape(param_var[i][j], newshape=(7, 3))
            #         # print(np.shape(cable_other_side))
            #         v1 = Value.cost_feasible_points(cable_one_side, cable_other_side, cable_length)
            #         v2 = Value.cost_cable_interference(cable_one_side, cable_other_side)
            #         v3 = Value.cost_cable_length(cable_one_side, cable_other_side)
            #         # r1 = Value.r_t_AW(cable_one_side, cable_other_side)
            #         # r2 = Value.r_r_AW(cable_one_side, cable_other_side)
            #         # v4 = r1+r2
            #         y_value = v1 + v2 + v3
            #         x_value = np.concatenate((x[i][j], param_var[i][j]))
            #         x_list.append(x_value)
            #         y_list.append(y_value)
            #     train_x.append(x_list)
            #     train_y.append(y_list)
            # train_x = np.array(train_x)
            # train_y = np.array(train_y)
            # sio.savemat('/home/chh3213/ros_wc/src/plan_cdpr/scripts/data' + '/process_data.mat',
            #             {'train_x': train_x, 'train_y': train_y})
            print('处理完毕')
            train_data = sio.loadmat('/home/chh3213/ros_wc/src/plan_cdpr/scripts/data/process_data.mat')
            train_x = train_data['train_x']
            train_y = train_data['train_y']

        '''++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
        '''开始train 部分'''
        net = Neuro_net(42, 32, 1).to(device)
        # optimizer 优化
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001,momentum=0.8,weight_decay=0.01)
        # 每跑50000个step就把学习率乘以0.9
        scheduler = StepLR(optimizer, step_size=10000, gamma=0.8)
        # loss funaction
        loss_funaction = torch.nn.MSELoss()
        step = 0

        '''++++++++++++++++++++++++++++++'''
        for j in range(epoches):
            # print(x[i])
            # print(param_var[i])
            for i in range(999):
                step +=1
                x_data = torch.tensor(train_x[i], device=device, dtype=torch.float32)
                y_data = torch.tensor(train_y[i], device=device, dtype=torch.float32)
                pridect_y = net(x_data)  # 喂入训练数据 得到预测的y值
                loss = loss_funaction(pridect_y, y_data)  # 计算损失
                optimizer.zero_grad()    # 为下一次训练清除上一步残余更新参数
                loss.backward()          # 误差反向传播，计算梯度
                optimizer.step()         # 将参数更新值施加到 net 的 parameters 上
                scheduler.step()
                if step % 100 == 0:
                    # print("已训练{}步 | loss：{} | y_data:{} | predict_y:{}.".format(i, loss, y_data.item().sum(), pridect_y.item().sum()))
                    print("已训练{}步 | loss：{} .".format(step, loss))
        torch.save(net, '/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/torch_fit.pt')
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
        x = np.random.random((64, 21)) * 3
        param_var = np.random.random((64, 21)) * 3

        cable_length = np.array([4.5 for _ in range(7)])
        net = Neuro_net(42, 64, 1).to(device)
        net = torch.load('/home/chh3213/ros_wc/src/plan_cdpr/scripts/model/torch_fit.pt')
        for i in range(len(x)):
            cable_other_side = np.reshape(x[i], newshape=(7, 3))
            cable_one_side = np.reshape(param_var[i], newshape=(7, 3))
            # print(np.shape(cable_other_side))
            v1 = Value.cost_feasible_points(cable_one_side,cable_other_side,cable_length)
            v2 = Value.cost_cable_interference(cable_one_side,cable_other_side)
            test_y = v1+v2
            test_x = np.concatenate((x[i], param_var[i]))
            '''2环境获取的数据'''
            x_data = torch.tensor(test_x, device=device, dtype=torch.float32,requires_grad=True)
            y_data = torch.tensor(test_y, device=device, dtype=torch.float32)
            y_ = net(x_data)
            y_.backward()  # 反向传播计算梯度
            grads_1 = x_data.grad
            print('+++++++++++++++')
            print('target', test_y)
            print('approximate', y_.item())
            # 梯度
            # print('torch_grads', grads_1)
            # print('func_grads', grads_2)
            # 误差
            print('func_mse: {}'.format(F.mse_loss(y_data, y_)))
            print('===================')
