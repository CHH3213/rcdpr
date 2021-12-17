# -*-coding:utf-8-*-
"""
使用神经网络求梯度--》得到下一步的uav状态
"""
from __future__ import print_function, absolute_import, division
from torch.autograd import Variable
import random
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from numpy.core.fromnumeric import shape
import torch
import scipy.io as sio
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, n_input=42, n_hidden1=32, n_hidden2=32, n_hidden3=16, n_output=1):
        super(Net, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_input, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_hidden3),
            nn.ReLU(),
            nn.Linear(n_hidden3, n_output),
        )

    def forward(self, input):
        return self.nn(input)


class Gradient_calculation:
    def __init__(self):
        # TODO: Net的参数设置
        self.Net = Net(11, 256,256,256, 1).to(device)
        self.alpha = 0.05# 0.2半径，300steps  2*pi/300*0.2
        print('======load model=========')
        self.Net.load_state_dict(torch.load(
            'figure/g/17/torch_fit_value2.pt'))
        # self.Net.load_state_dict(torch.load(
        #     '/home/firefly/cable_fit/save/model/torch_fit_value2.pt'))
        

    def torch_nn(self, x):
        """
        使用pytorch 得到的神经网络求梯度
        :param x: 变量x，14维度  uav的x,y，platform的x,y,z和r,p,y
        :return: 下一步的x,只需要uav的x,y，共8维
        """
        x_data = torch.tensor(
            x, device=device, dtype=torch.float32, requires_grad=True)
        y_ = self.Net(x_data)
        y_.backward()  # 反向传播计算梯度
        grads_1 = x_data.grad  # 测试梯度
        grads = grads_1.tolist()
        grads_temp = grads[0:8]
        # if np.linalg.norm(grads_temp)>1:
        print('y', y_.item())
        # print('grads_temp', grads_temp)
        # print('np.linalg.norm(grads_temp)', np.linalg.norm(grads_temp))
        if np.linalg.norm(grads_temp) != 0:
            grads_temp = grads_temp/np.linalg.norm(grads_temp)  # 归一化 
        # print('grads_temp', grads_temp)
        # print('grads_temp', np.array(grads_temp))


        x_now = x_data.tolist()
        x_next = np.array(x_now[0:8])-np.array(grads_temp)*self.alpha
        # print(x_next)

        # x_test = np.concatenate((x_next, x_now[8:]), axis=0)
        # x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
        # y_test = self.Net(x_test)
        # print('real x:{}', x)
        # print('x: ', x_test)
        # print('y:{}\ty_next:{}'.format(y_.item(), y_test.item()))

        # if y_.item() > 2000:
        #     x_next = None

        # if y_test.item() > y_.item():
        #     return np.array(x_now[0:8])
            
        return x_next
