# -*-coding:utf-8-*-
'''
获取训练数据
'''
import os
import sys

sys.path.append('..')
sys.path.append('/home/chh3213/ros_wc/src/plan_cdpr/scripts')
from value_function import Value_function

sys.path.append('../scripts/data')
import numpy as np
import scipy.io as sio


def train_data(is_random=True, is_random_load=True, fit_v_all=False, fit_v1=False, fit_v2=False, fit_v3=False,
               fit_v4=False):
    if not is_random:
        '''数据准备'''
        FileName = '/home/chh3213/ros_wc/src/plan_cdpr/scripts/data/value_data.mat'
        train_data = sio.loadmat(FileName)
        # dict_keys(['__header__', '__version__', '__globals__', 'input_platform', 'input_pos', 'output_value'])
        # print(train_data.keys())
        # input_pos：uav1~4, ugv1~3  input_platform：platform_point
        print(np.shape(train_data['input_pos']))
        # print(train_data['input_pos'][0])
        x = []
        number = 1999-500
        for i in range(500, 1999):
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
        train_y = np.array(train_data['output_value'][500:2000])
        train_y = np.reshape(train_y, (len(train_y), 1))
        print(np.shape(train_x))
        print(np.shape(train_y))
        # for k in range(0,100):
        #     print(train_y[k])
    else:
        '''################随机生成数据训练#####################'''
        v_data = 'process_data'
        number = 10000
        if fit_v_all:
            v_data = 'process_data'
            fit_v1, fit_v2, fit_v3, fit_v4 = True, True, True, True
        if fit_v1:
            v_data = 'v1_data'
        if fit_v2:
            v_data = 'v2_data'
        if fit_v3:
            v_data = 'v3_data'
        if fit_v4:
            v_data = 'v4_data'
        if not is_random_load:
            Value = Value_function()
            # np.random.seed(1)
            x = np.random.random((number, 21)) * 6 - 3

            param_var = np.random.random((number, 21)) * 6 - 3
            # print(shape(x))
            train_x = []
            train_y = []
            cable_length = np.array(
                [5 for _ in range(7)])  # FIXME: 改为5， 原来是4.5
            rotation_center = np.array([0, 0, 3])  # FIXME: 增加， r_r_AW 需要，但原来的没有


            for i in range(number):
                '''1随机的数据--需要做的处理'''
                # print('processing ', i)
                cable_other_side = np.reshape(x[i], newshape=(7, 3))
                cable_one_side = np.reshape(param_var[i], newshape=(7, 3))
                # print(np.shape(cable_other_side))
                v1 = Value.cost_feasible_points(cable_one_side, cable_other_side, cable_length) if fit_v1 else 0
                v2 = Value.cost_cable_interference(cable_one_side, cable_other_side) if fit_v2 else 0
                v3 = Value.cost_cable_length(cable_one_side, cable_other_side) if fit_v3 else 0
                r1 = Value.r_t_AW(cable_one_side, cable_other_side)
                r2 = Value.r_r_AW(cable_one_side, cable_other_side, rotation_center)
                v4 = r1 + r2 if fit_v4 else 0

                y_value = v1 + v2+v3+v4
                x_value = np.concatenate((x[i], param_var[i]))
                train_x.append(x_value)
                train_y.append(y_value)
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            train_y = train_y.reshape((number, 1))
            print(np.shape(train_y))
            print(np.shape(train_x))
            sio.savemat('/home/chh3213/ros_wc/src/plan_cdpr/scripts/data' + '/'+v_data+'.mat',
                        {'train_x': train_x, 'train_y': train_y})
            print('处理完毕')
            # print('train_x ', train_x)
            # print('train_y ', train_y)
        else:
            train_data = sio.loadmat('/home/chh3213/ros_wc/src/plan_cdpr/scripts/data'+ '/'+v_data+'.mat')
            train_y = train_data['train_y']
            train_x = train_data['train_x']
            print(np.shape(train_x))
            print(np.shape(train_y))
    return train_x, train_y, number

def exper_data(is_random=True, is_random_load=True, fit_v_all=False, fit_v1=False, fit_v2=False, fit_v3=False,
               fit_v4=False):
    if not is_random:
        '''数据准备'''
        FileName = '/home/chh3213/ros_wc/src/plan_cdpr/scripts/data/value_data.mat'
        test_data = sio.loadmat(FileName)
        # dict_keys(['__header__', '__version__', '__globals__', 'input_platform', 'input_pos', 'output_value'])
        # print(test_data.keys())
        # input_pos：uav1~4, ugv1~3  input_platform：platform_point
        print(np.shape(test_data['input_pos']))
        # print(test_data['input_pos'][0])
        x = []
        number = 1999-500
        for i in range(500, 1999):
            T1 = np.array(test_data['input_pos']
                          [i]).flatten()  # 注意数据是否已经展平过
            T2 = np.array(test_data['input_platform'][i]).flatten()
            T_platform = np.concatenate((T1, T2))  # 前21为是位置变量，后21位是平台固定参数
            x.append(T_platform)
            # print(np.shape(T_platform))
        test_x = np.array(x)
        print(np.shape(x))
        # print(x[:, 0])
        # print(type(x))  # uav1~4, ugv1~3
        # print(np.shape(x))
        test_data['output_value'] = np.squeeze(test_data['output_value'])
        # print(np.shape(test_data['output_value']))
        test_y = np.array(test_data['output_value'][500:2000])
        test_y = np.reshape(test_y, (len(test_y), 1))
        print(np.shape(test_x))
        print(np.shape(test_y))
        # for k in range(0,100):
        #     print(test_y[k])
    else:
        '''################随机生成数据训练#####################'''
        v_data = 'process_data'
        number = 128
        if fit_v_all:
            v_data = 'process_data'
            fit_v1, fit_v2, fit_v3, fit_v4 = True, True, True, True
        if fit_v1:
            v_data = 'v1_data'
        if fit_v2:
            v_data = 'v2_data'
        if fit_v3:
            v_data = 'v3_data'
        if fit_v4:
            v_data = 'v4_data'
        Value = Value_function()
        # np.random.seed(1)
        x = np.random.random((number, 21)) * 6 - 3

        param_var = np.random.random((number, 21)) * 6 - 3
        # print(shape(x))
        test_x = []
        test_y = []
        cable_length = np.array(
            [5 for _ in range(7)])  # FIXME: 改为5， 原来是4.5
        rotation_center = np.array([0, 0, 3])  # FIXME: 增加， r_r_AW 需要，但原来的没有


        for i in range(number):
            '''1随机的数据--需要做的处理'''
            # print('processing ', i)
            cable_other_side = np.reshape(x[i], newshape=(7, 3))
            cable_one_side = np.reshape(param_var[i], newshape=(7, 3))
            # print(np.shape(cable_other_side))
            v1 = Value.cost_feasible_points(cable_one_side, cable_other_side, cable_length) if fit_v1 else 0
            v2 = Value.cost_cable_interference(cable_one_side, cable_other_side) if fit_v2 else 0
            v3 = Value.cost_cable_length(cable_one_side, cable_other_side) if fit_v3 else 0
            r1 = Value.r_t_AW(cable_one_side, cable_other_side)
            r2 = Value.r_r_AW(cable_one_side, cable_other_side, rotation_center)
            v4 = r1 + r2 if fit_v4 else 0

            y_value = v1 + v2+v3+v4
            x_value = np.concatenate((x[i], param_var[i]))
            test_x.append(x_value)
            test_y.append(y_value)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_y = test_y.reshape((number, 1))
        print(np.shape(test_y))
        print(np.shape(test_x))
    return test_x, test_y



if __name__ == '__main__':
    train_data(is_random=True, is_random_load=True,fit_v2=True)
