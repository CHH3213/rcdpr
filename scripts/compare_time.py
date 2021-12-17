import scipy.io as sio
from opt_scipy_value import minimize_cost
from value_function import Value_function
import numpy as np
import time
from gradient_calculation import Gradient_calculation
import matplotlib.pyplot as plt

value = Value_function()
save_dir = 'figure/g/17/net_small2/'

def myplot(ax, data, labels= None):
    ax.plot(data, label=labels)
    ax.grid()
    ax.legend()


def temp_func(Filename):
    data = sio.loadmat(save_dir+Filename)
    opt_nexts = data['opt_nexts']
    net_nexts = data['net_nexts']
    deltas = data['deltas']

    opt_nexts = np.array(opt_nexts).T
    net_nexts = np.array(net_nexts).T
    deltas = np.array(deltas).T

    mse = []
    
    for i in range(np.shape(deltas)[1]):
        if i%10==0:
            mse.append(np.linalg.norm(deltas[0:2, i]) +
                       np.linalg.norm(deltas[2:4, i]) +
                       np.linalg.norm(deltas[4:6, i]) +
                       np.linalg.norm(deltas[6:8, i]))



    datas = [opt_nexts, net_nexts, deltas, mse]
    figs = []
    axs = []
    for i in range(2):
        fig, ax = plt.subplots(2, 4)
        figs.append(fig)
        axs.append(ax)

    fig, ax = plt.subplots(1)
    figs.append(fig)
    axs.append(ax)

    fig_title = ['opt and net', 'delta', 'mean square error']
    for fig_index in range(3):
        figs[fig_index].suptitle(fig_title[fig_index])
        # for ax_index in range(2):
        if fig_index == 2:
            myplot(axs[fig_index], mse, 'mse')
        else:
            for i in range(2):
                for j in range(4):
                    if fig_index == 0:
                        myplot(axs[fig_index][i][j], datas[fig_index][4*i+j, :], 'opt')
                        myplot(axs[fig_index][i][j], datas[fig_index+1][4*i+j, :], 'net')
                    else:
                        myplot(axs[fig_index][i][j], datas[fig_index+1][4*i+j, :], 'delta')

    plt.show()


def optimizer_net_time(Filename):
    value_input = sio.loadmat(save_dir+Filename)
    print(len(value_input['CABLE_ONE_SIDE']))

    # 初始化神经网络
    grad = Gradient_calculation()
    x = np.array([-1.5, 1.5, -1.5, -1.5, 1.5, -1.5, 1.5, 1.5,0.0,0.0,4.0])
    net_next = grad.torch_nn(x)
    
    # constant parameters
    x_ugv = np.array([2.5, 1.45, 0.06, -2.5, 1.45, 0.06, 0, -2.9, 0.06])
    m, d_min, mu_l, mu_t, mu_r, mu_d, mu_a, target_length_list, t_min, t_max =\
        value.m, value.d_min, value.mu_l, value.mu_t, value.mu_r, value.mu_d, value.mu_a,value.target_length_list, value.t_min, value.t_max
    t_min, t_max = np.array(t_min.T[0]), np.array(t_max.T[0])
    point_end_effector = np.array([[0., 0., 0.] for i in range(7)])
    opt_times = []
    opt_nexts = []
    net_nexts = []
    net_times = []
    deltas = []

    for i in range(0, len(value_input['CABLE_ONE_SIDE'])):
        
        CABLE_ONE_SIDE = value_input['CABLE_ONE_SIDE'][i]
        CABLE_OTHER_SIDE = value_input['CABLE_OTHER_SIDE'][i]
        rotation_cent = value_input['ROTATION_CENTER'][i]
        pose_0 = value_input['POSE_0'][i]

        uav_pos = CABLE_OTHER_SIDE[:4, :2].flatten()
        # print(CABLE_ONE_SIDE)
        # print(x_ugv)
        # print(m, d_min, mu_l, mu_t,mu_r, mu_d, mu_a)
        # print(rotation_cent)
        # print(target_length_list)
        # print(t_min, t_max)
        # print(uav_pos)
        # print(point_end_effector)
        # print(pose_0)
        

        args = (CABLE_ONE_SIDE, x_ugv, m, d_min, mu_l, mu_t,
                mu_r, mu_d, mu_a, rotation_cent,
                target_length_list, t_min, t_max, uav_pos, point_end_effector, pose_0)
        
        opt_start = time.time()
        opt_next = minimize_cost(args)
        opt_times.append(time.time() - opt_start)
        opt_nexts.append(opt_next)

        net_input = np.concatenate((uav_pos, rotation_cent))
        net_start_time = time.time()
        net_next = grad.torch_nn(net_input)
        net_times.append(time.time() - net_start_time)
        net_nexts.append(net_next)

        delta_a = net_next - opt_next
        deltas.append(delta_a)
        # print(delta_a)

        print('No.{}\ttime:{}'.format(i, opt_times[-1]))

    sio.savemat(save_dir + 'opt_net_time_next.mat',
                dict(opt_times=opt_times, opt_nexts=opt_nexts, net_times=net_times, net_nexts=net_nexts, deltas=deltas))

    opt_nexts = np.array(opt_nexts).T
    net_nexts = np.array(net_nexts).T
    deltas = np.array(deltas).T

    datas = [opt_nexts, net_nexts, deltas]
    figs = []
    axs = []
    for i in range(3):
        fig, ax = plt.subplots(2, 4)
        figs.append(fig)
        axs.append(ax)

    fig_title = ['opt', 'net', 'delta']
    for fig_index in range(3):
        figs[fig_index].suptitle(fig_title[fig_index])
        # for ax_index in range(3):
        for i in range(2):
            for j in range(4):
                myplot(axs[fig_index][i][j], datas[fig_index][4*i+j, :])
    
    plt.show()






if __name__ == '__main__':
    # Filename = 'network_value_input_02_0.mat'
    # optimizer_net_time(Filename)
    Filename = 'opt_net_time_next.mat'
    temp_func(Filename)




