#!/usr/bin/python3
# -*-coding:utf-8-*-
# @Project: ugv_uav
# @File: torch_fitting.py---使用torch 神经网络对函数拟合求梯度
# @Author: CHH
# @Time: 2021/8/24 上午10:57
'''
    训练完了，发现隐层越大，拟合的速度越是快，拟合的效果越是好
'''

from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import torch
import scipy.io as sio
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

class Neuro_net(torch.nn.Module):
    """搭建神经网络"""

    def __init__(self, n_feature=21, n_hidden=128, n_output=1):
        super(Neuro_net, self).__init__()   # 继承__init__功能
        self.hidden_layer1 = torch.nn.Linear(n_feature, n_hidden)
        # self.hidden_layer2 = torch.nn.Linear(n_hidden, 32)
        self.output_layer = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = F.relu(x)
        # x = self.hidden_layer2(x)
        # x = F.relu(x)

        pridect_y = self.output_layer(x)
        return pridect_y

if __name__ == '__main__':

    train=True
    # train=False

    if train ==True:
        '''数据准备'''
        FileName = './scripts/data/train_data.mat'
        train_data = sio.loadmat(FileName)
        # dict_keys(['__header__', '__version__', '__globals__', 'timestep', 'pos'])
        # print(train_data.keys())
        # # (1500, 7, 3) payload,drone1,drone2,car0,car1,car2,car3
        # print(np.shape(train_data['pos']))
        # print(train_data['pos'][0])
        x = []
        for i in range(500, 1200):
            T = np.array(train_data['pos'][i]).flatten()
            x.append(T)
        x = np.array(x)
        # print(np.shape(x[:3]))
        # print(x[:, 0])
        print(type(x))  # payload,drone1,drone2,car0,car1,car2,car3
        print(np.shape(x))
        # 开始500步为稳定阶段，对训练无效
        d1, d2 = np.linalg.norm(
            [0, -3, 4.5]), np.linalg.norm([0, 3, 4.5])  # 固定距离
        d3, d4, d5, d6 = np.linalg.norm([5, 5, 3]), np.linalg.norm([5, 5, 3]), np.linalg.norm(
            [5, 5, 3]), np.linalg.norm([5, 5, 3])  # 固定距离
        d_xy = 5 * np.sqrt(2)
        p0 = x[:, 0:3]
        p0_xy = x[:, 0:2]
        train_y = []
        for i in range(len(x)):
            y: object = np.abs(np.linalg.norm(p0[i] - x[i, 3:6]) - d1) + np.abs(np.linalg.norm(p0[i] - x[i, 6:9]) - d2) + np.abs(
                np.linalg.norm(p0[i] - x[i, 9:12]) - d3) + \
                np.abs(np.linalg.norm(p0[i] - x[i, 12:15]) - d4) + np.abs(np.linalg.norm(p0[i] - x[i, 15:18]) - d5) + np.abs(
                np.linalg.norm(p0[i] - x[i, 18:]) - d6)
            train_y.append(y)
        train_y = np.array(train_y).reshape(len(train_y), 1)
        # print(np.shape(train_y))
        # '''++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
        '''开始train 部分'''
        net = Neuro_net(21, 128, 1)
        # optimizer 优化
        optimizer = torch.optim.SGD(net.parameters(), lr=0.005,momentum=0.78,weight_decay=0.0)
        # 每跑50000个step就把学习率乘以0.9
        scheduler = StepLR(optimizer, step_size=50000, gamma=0.99)
        # loss funaction
        loss_funaction = torch.nn.MSELoss()
        epoch = 500000
        x_data = torch.tensor(x, device=device, dtype=torch.float32)
        y_data = torch.tensor(train_y, device=device, dtype=torch.float32)
        for step in range(epoch):
            pridect_y = net(x_data)  # 喂入训练数据 得到预测的y值
            loss = loss_funaction(pridect_y, y_data)  # 计算损失

            optimizer.zero_grad()    # 为下一次训练清除上一步残余更新参数
            loss.backward()          # 误差反向传播，计算梯度
            optimizer.step()         # 将参数更新值施加到 net 的 parameters 上
            scheduler.step()
            if step % 100 == 0:
                print("已训练{}步 | loss：{}.".format(step, loss))
        torch.save(net, '../scripts/model/torch_model.pt')
    else:
        # test
        # 过拟合
        d1, d2 = np.linalg.norm(
            [0, -3, 4.5]), np.linalg.norm([0, 3, 4.5])  # 固定距离
        d3, d4, d5, d6 = np.linalg.norm([5, 5, 3]), np.linalg.norm([5, 5, 3]), np.linalg.norm(
            [5, 5, 3]), np.linalg.norm([5, 5, 3])  # 固定距离
        d_xy = 5 * np.sqrt(2)
        '''================数据准备========================'''
        FileName = '../scripts/data/train_data.mat'
        train_data = sio.loadmat(FileName)
        # dict_keys(['__header__', '__version__', '__globals__', 'timestep', 'pos'])
        print(train_data.keys())
        # (1500, 7, 3) payload,drone1,drone2,car0,car1,car2,car3
        print(np.shape(train_data['pos']))
        print(train_data['pos'][0])
        x_test = []
        for i in range(1100, 1500):
            T = np.array(train_data['pos'][i]).flatten()
            x_test.append(T)
        x_test = np.array(x_test)
        # 随机生成数据
        # x_test = np.array([(2*np.random.random(200)-1)*1 for _ in range(21)])
        # x_test = x_test.T
        # print(np.shape(x_test))
        p0 = x_test[:, 0:3]
        test_y = []
        for i in range(len(x_test)):
            y: object = np.abs(np.linalg.norm(p0[i] - x_test[i, 3:6]) - d1) + np.abs(np.linalg.norm(p0[i] - x_test[i, 6:9]) - d2) + np.abs(
                np.linalg.norm(p0[i] - x_test[i, 9:12]) - d3) + \
                np.abs(np.linalg.norm(p0[i] - x_test[i, 12:15]) - d4) + np.abs(np.linalg.norm(p0[i] - x_test[i, 15:18]) - d5) + np.abs(
                np.linalg.norm(p0[i] - x_test[i, 18:]) - d6)
            test_y.append(y)
        test_y = np.array(test_y).reshape(len(test_y), 1)
        '''#############################################'''

        net = torch.load('../scripts/model/torch_model.pt')
        for i in range(len(test_y)):
            x_ = torch.tensor(x_test[i], dtype=torch.float32,device=device,requires_grad=True)
            y_ = net(x_)
            y_.backward()  # 反向传播计算梯度
            grads_1 = x_.grad

            x = torch.tensor(x_[3:],dtype=torch.float32,device=device,requires_grad=True)
            p0 = x_[0:3]
            # 目标函数
            y: object = torch.abs(torch.norm(p0-x[0:3])-d1)+torch.abs(torch.norm(p0-x[3:6])-d2)+torch.abs(torch.norm(p0-x[6:9])-d3) + \
                torch.abs(torch.norm(p0-x[9:12])-d4)+torch.abs(torch.norm( p0 - x[12:15])-d5)+torch.abs(torch.norm(p0-x[15:])-d6)

            y.backward()  # 反向传播计算梯度
            grads_2 = x.grad
            # 目标值
            print('target', test_y[i])
            print('approximate', y_.detach().numpy())
            # 梯度
            # print('torch_grads', grads_1[3:])
            # print('func_grads', grads_2)
            # 误差
            print('grad_mse: {}'.format(F.mse_loss(grads_1[3:], grads_2)))
            print('func_mse: {}'.format(F.mse_loss(y, y_)))
