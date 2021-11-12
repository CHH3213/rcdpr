#!/usr/bin/python3
# -*-coding:utf-8-*-
# @Project: plan_cdpr
# @File: torch_fit_v6.py---输入降维度，14维，不考虑小车，小车固定，平台使用质心位置+姿态，无人机不关系高度
# @Author: CHH
# @Time: 2021/11/4 下午21：39
'''
    训练完了，发现隐层越大，拟合的速度越是快，拟合的效果越是好
'''
from __future__ import print_function, absolute_import, division
from torch.autograd import Variable
import random
import sys
import matplotlib.pyplot as plt
import tf

# sys.path.append('..')
# sys.path.append('/home/chh3213/ros_wc/src/plan_cdpr/scripts')
from value_function import Value_function

# sys.path.append('../scripts/data')
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
import copy

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cuda'


class DataPrefetcher():
    '''
    TODO: 数据预处理，可能存在显存溢出的问题，见https://github.com/NVIDIA/apex/issues/439

    '''
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        # with torch.cuda.stream(self.stream):
        #     for k in self.batch:
        #         
        #         if k != 'meta':
        #             self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch
    
    def reset(self, loader):
        '''
        TODO: 重新加载数据集，在一个episode结束时使用，用于下一个episode
        '''
        self.loader = iter(loader)
        self.preload()


# device = "cpu"
class Net(nn.Module):
    def __init__(self, n_input=42, n_hidden1=32, n_hidden2=32,n_hidden3=16, n_output=1):
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


def env_generate():
    '''数据准备'''
    FileName = 'save/data/value_data_5.mat'
    train_data = sio.loadmat(FileName)
    # dict_keys(['__header__', '__version__', '__globals__', 'input_platform', 'input_pos', 'output_value'])
    # print(train_data.keys())
    # input_pos：uav1~4, ugv1~3  input_platform：platform_point
    # print(np.shape(train_data['input_pos']))
    # print(train_data['output_value'])
    number = len(train_data['input_pos'])-500
    x = []
    for i in range(500, len(train_data['input_pos'])):
        T1 = np.array(train_data['input_pos'][i]).flatten()  # 注意数据是否已经展平过
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
    train_y = np.array(train_data['output_value']
                       [500:len(train_data['input_pos'])])
    train_y = np.reshape(train_y, (len(train_y), 1))
    print(np.shape(train_x))
    print(np.shape(train_y))
    return train_x, train_y, number


def random_generate(number, is_test=False):
    # np.random.seed(1)
    cable_length = np.array([5 for _ in range(7)]) 
    train_x = []
    train_y = []
    x_uav = np.random.random((number, 8)) * 8- 4  # 4 uav, includes x,y
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
    # x_uav_temp = [x_uav_1_x,x_uav_1_y,x_uav_2_x,x_uav_2_y,x_uav_3_x,x_uav_3_y,x_uav_4_x,x_uav_4_y]
    # x_uav_ = []
    # for i in range(8):
    #     uav_temp = x_uav_temp[i]
    #     # print(len(uav_temp))
    #     np.random.shuffle(uav_temp)
    #     x_uav_.append(copy.deepcopy(uav_temp))
    # # print(x_uav_)
    # x_uav = np.array(x_uav_).T


    x_ugv = np.array([2.5, 1.45, 0, -2.5, 1.45, 0, 0, -2.9, 0])  # 9

    #######random
    # param_var = np.random.random((number, 6)) * 2-1 # platform pose x,y,z,r,p,y
    param_var_x = np.random.random((96000))*2-1
    param_var_y = np.random.random((96000))*2-1
    param_var_z = np.random.random((96000))*2-1+4
    param_var_euler = np.random.random((96000))*2-1
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
    param_var_temp = [param_var_x,param_var_y,param_var_z,param_var_euler,copy.deepcopy(param_var_euler),copy.deepcopy(param_var_euler)]
    param_var_ = []
    for i in range(6):
        temp = param_var_temp[i]
        # print('r',len(temp))
        temp = temp[:96000]
        np.random.shuffle(temp)
        param_var_.append(copy.deepcopy(temp))
    param_var = np.array(param_var_).T

    print(np.shape(x_uav))
    print(np.shape(param_var))

    # x_uav = np.array([[-2.39837861 , 2.0151034 ,  0.039017 ,  -2.27771882 , 2.56088251 , 2.00710349,0.05582332,  0.54714831]])
    # param_var = np.array([[0.05360465 , 0.5804346 ,  4.22081464 , 0.20645114,-1.12417206 , 2.32108617]])

    # 末端执行器固定点在其自身坐标系中的位置uav1~4,ugv1~3
    point_end_effector = np.array(
        [np.array([0.0, 0.29, 0.25]), np.array([-0.25, -0.145, 0.25]),
         np.array([0.25, -0.145, 0.25]), np.array([0.0, 0.0, 0.25]),
         np.array([0.0, 0.29, -0.25]), np.array([-0.25, -0.145, -0.25]),
         np.array([0.25, -0.145, -0.25])])
    if not is_test:
        if not True:
        # if os.path.exists('save/data/fit_version6_v2_data.mat'):
            train_data = sio.loadmat('save/data/fit_version6_v2_data.mat')
            train_y = train_data['train_y']
            train_x = train_data['train_x']
        else:
            for i in range(np.shape(x_uav)[0]):
                if i % 10 == 0:
                    print(i)
                x_uav_temp = np.insert(x_uav[i], [2, 4, 6, 8], [
                                       7.5, 7.5, 7.5, 7.5])  # 补足uav位置信息x,y,z
                cable_other_side = np.concatenate((x_uav_temp, x_ugv))
                cable_other_side = np.reshape(
                    cable_other_side, newshape=(7, 3))  # agent pos
                rot_center = np.array(param_var[i, 0:3])  # 末端执行器坐标系相对于基坐标系的位置
                pose_0 = tf.transformations.quaternion_from_euler(
                    *param_var[i, 3:6])  # 四元数
                cable_one_side = np.empty((7, 3))  # 7个固定点，每个点是 3惟的
                for j in range(0, 7, 1):
                    vector = np.hstack([point_end_effector[j], 0])  # 将B点转换为四元数
                    # 通过四元数的旋转变换求得旋转后B’的四元数位置（相对于末端执行器坐标系）
                    rotated_vector = tf.transformations.quaternion_multiply(pose_0,
                                                                            tf.transformations.quaternion_multiply(
                                                                                vector,
                                                                                tf.transformations.quaternion_inverse(pose_0)))
                    # 将B’的四元数位置变成3维位置并进行坐标补偿将它变成相对于基坐标系的位置
                    cable_one_side[j] = np.delete(
                        rotated_vector, 3) + rot_center
                Value.set_jacobian_param(point_end_effector,pose_0,rot_center)
                v1 = Value.cost_feasible_points(cable_one_side, cable_other_side, cable_length)
                v2 = Value.cost_cable_interference(cable_one_side, cable_other_side)
                # v3 = Value.cost_cable_length(cable_one_side, cable_other_side)
                r1 = Value.r_t_AW(cable_one_side, cable_other_side)
                r2 = Value.r_r_AW(cable_one_side, cable_other_side, rot_center)
                v4 = r1+r2
                y_value = v1+v2+v4
                # print('y_value:{}\t interferece: {}\t feasible: {}\t r: {}'.format(y_value,v2,v1,v4))
                x_value = np.concatenate((x_uav[i], param_var[i]))  # 14dim
                train_x.append(x_value)
                train_y.append(y_value)
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            train_y = train_y.reshape((np.shape(x_uav)[0], 1))
            print(np.shape(train_y))
            print(np.shape(train_x))
            sio.savemat('/home/firefly/chh_ws/src/plan_cdpr/scripts/data/-5_5_-2_2_96000.mat',
                        {'train_x': train_x, 'train_y': train_y})
        print('处理完毕')
        print('train_x shape ', np.shape(train_x))
        print('train_y shape', np.shape(train_y))
    else:
        for i in range(128):
            x_uav_temp = np.insert(x_uav[i], [2, 4, 6, 8], [
                                  8,8,8,8])  # 补足uav位置信息x,y,z
            # print(x_ugv[i])
            cable_other_side = np.concatenate((x_uav_temp, x_ugv))
            cable_other_side = np.reshape(
                cable_other_side, newshape=(7, 3))  # agent pos
            rot_center = np.array(param_var[i, 0:3])  # 末端执行器坐标系相对于基坐标系的位置
            pose_0 = tf.transformations.quaternion_from_euler(
                *param_var[i, 3:6])  # 四元数
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

            v1 = Value.cost_feasible_points(cable_one_side, cable_other_side, cable_length)
            v2 = Value.cost_cable_interference(
                cable_one_side, cable_other_side)
            # v3 = Value.cost_cable_length(cable_one_side, cable_other_side)
            r1 = Value.r_t_AW(cable_one_side, cable_other_side)
            r2 = Value.r_r_AW(cable_one_side, cable_other_side, rot_center)
            v4 = r1+r2
            y_value = v2+v1+v4
            # print('y_value', y_value)
            x_value = np.concatenate((x_uav[i], param_var[i]))  # 14dim
            train_x.append(x_value)
            train_y.append(y_value)
        train_x = np.array(train_x)
        train_y = np.array(train_y)

        train_y = train_y.reshape(number, 1)
        print(np.shape(train_y))
        print(np.shape(train_x))

    return train_x, train_y


if __name__ == '__main__':
    Value = Value_function()
    random_data = True
    train = True
    # train = False
    if train == True:
        epoches = 3600000
        if random_data == False:
            train_x, train_y, number = env_generate()
        else:
            '''################随机生成数据训练#####################'''
            number = 96000
            train_x, train_y = random_generate(number, False)

    #     train_x = torch.tensor(train_x, device=device, dtype=torch.float32)
    #     train_y = torch.tensor(train_y, device=device, dtype=torch.float32)
    #     for i in range(100):
    #         print('x: {} \n y:{}'.format(train_x[i], train_y[i]))
    #     # ds = torch.utils.data.TensorDataset(train_x, train_y)
    #     # loader = torch.utils.data.DataLoader(
    #     #     ds, batch_size=5000, shuffle=True, pin_memory=True)
    #     # prefetcher = DataPrefetcher(loader=loader, device=device)
    #     '''++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
    #     '''开始train 部分'''
    #     net = Net(14, 512,512,256, 1).to(device)
    #     # optimizer 优化
    #     if os.path.exists('save/model/model_14dim/torch_fit_value2.pt'):
    #         print('load model!!!')
    #         print('**********************')
    #         net.load_state_dict(torch.load(
    #             'save/model/model_14dim/torch_fit_value2.pt'))
    #     optimizer = torch.optim.SGD(net.parameters(), lr=1e-9,weight_decay=0.1)
    #     net = nn.DataParallel(net)
    #     # 每跑50000个step就把学习率乘以0.9
    #     # scheduler = ReduceLROnPlateau(optimizer, 'min',patience=100,factor=0.5)
    #     # scheduler = StepLR(optimizer, step_size=6000, gamma=0.8)
    #     # loss funaction
    #     loss_funaction = torch.nn.MSELoss()
    #     step = 0

    #     # 尝试
    #     coun = 0
    #     start = time.time()
    #     for i in range(epoches):
    #         ##############################
    #         # batch = prefetcher.next()
    #         # while batch is not None:
    #         #     train_x, train_y = batch
    #         #     pridect_y = net(train_x)  # 喂入训练数据 得到预测的y值
    #         #     optimizer.zero_grad()  # 为下一次训练清除上一步残余更新参数
    #         #     loss = loss_funaction(pridect_y, train_y)  # 计算损失
    #         #     loss.backward()  # 误差反向传播，计算梯度
    #         #     optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
    #         #     scheduler.step()
    #         #     batch = prefetcher.next()
    #         #     print('loss',loss.item())
    #         # prefetcher.reset(loader)
    #         # batch = prefetcher.next()
    #         ################################################
    #         pridect_y = net(train_x)  # 喂入训练数据 得到预测的y值
    #         optimizer.zero_grad()  # 为下一次训练清除上一步残余更新参数
    #         loss = loss_funaction(pridect_y, train_y)  # 计算损失
    #         loss.backward()  # 误差反向传播，计算梯度
    #         optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
    #         # scheduler.step()
    #         #################################################
    #         # for batch_idx, (data, target) in enumerate(loader):
    #         #     data, target = Variable(data), Variable(target)
    #         #     pridect_y = net(data)  # 喂入训练数据 得到预测的y值
    #         #     optimizer.zero_grad()  # 为下一次训练清除上一步残余更新参数
    #         #     loss = loss_funaction(pridect_y, target)  # 计算损失
    #         #     loss.backward()  # 误差反向传播，计算梯度
    #         #     optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
    #         #     print('loss: ',loss.item())
    #         # if i <= 5000:
    #         #     scheduler.step()
    #         # scheduler.step()
    #         # if (batch_idx+1) % 100 == 0:
    #         #     print(batch_idx, loss.item())
    #         # pridect_y = net(Variable(train_x,requires_grad = False))

    #         if i % 100 == 0:
    #             end = time.time()
    #             # plt.cla()
    #             # plt.plot(train_y.cpu().numpy())
    #             # plt.plot(pridect_y.data.cpu().numpy(), 'r-')
    #             # plt.pause(0.3)
    #             # print("已训练{}步 | loss：{} | y_data:{} | predict_y:{}.".format(i, loss, y_data.item().sum(), pridect_y.item().sum()))
    #             print("已训练{}步 | loss：{} \t take {}s .".format(i, loss,end - start))
    #             start = end
    #             torch.save(
    #                 net.module.state_dict(), 'save/model/model_14dim/torch_fit_value2.pt')
    #     '''++++++++++++++++++++++++++++++'''

    #     torch.save(
    #         net.module.state_dict(), 'save/model/model_14dim/torch_fit_value2.pt')
    # else:
    #     '''#####################随机生成测试数据########################'''
    #     net = Net(14, 512,512,256, 1).to(device)
    #     net.load_state_dict(torch.load(
    #         'save/model/model_14dim/512512256-100000-11-8-9_30.pt'))
    #     plot_y = []
    #     plot_y_ = []
    #     plot_loss = []
    #     number = 128
    #     '''随机生成'''
    #     test_x, test_y = random_generate(number, True)
    #     '''环境获取'''
    #     # test_x, test_y, number = env_generate()
    #     # print(number)
    #     rloss = []
    #     for i in range(128):
    #         start_ = time.time()
            
    #         x_data = torch.tensor(test_x[i], device=device,
    #                               dtype=torch.float32, requires_grad=True)
    #         y_data = torch.tensor(
    #             test_y[i], device=device, dtype=torch.float32)
    #         y_ = net(x_data)
    #         # x_data.grad.zero_()
    #         y_.backward()  # 反向传播计算梯度
            
    #         grads_1 = x_data.grad
    #         related_loss = (y_.item() - test_y[i])/y_.item()
    #         rloss.append(related_loss)
    #         print('+++++++++++++++')
    #         print('x: ',x_data)
    #         print('target', test_y[i])
    #         print('approximate', y_.item())
    #         plot_y.append(test_y[i])
    #         plot_y_.append(y_.item())
    #         # 梯度
    #         print('torch_grads', grads_1)
    #         # print('func_grads', grads_2)
    #         # 误差
    #         print('func_mse: {}'.format(related_loss))
    #         plot_loss.append(F.mse_loss(y_data, y_).item())
    #         end_ = time.time()
    #         print('time: ',end_ - start_)
    #         print('===================')
        
    #     plt.title('predict')
    #     plt.plot(plot_y, 'b-')
    #     plt.plot(plot_y_, 'r-')
    #     plt.legend(['real_y', 'predict_y'])
    #     plt.figure()
    #     plt.plot(rloss)
    #     plt.legend(['loss'])
    #     plt.title('loss')
    #     # plt.ylim(top=10, bottom=-10)
    #     plt.show()
