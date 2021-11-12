from value_function import Value_function
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import time
from Jacobi import jacobi
value = Value_function()


# sm表示滑动窗口大小,为2*k+1, smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")

            smooth_data.append(d)

        return smooth_data
    return data

def calculate_r(Filename1, Filename2):
    print('processing ',Filename1)
    print('processing ',Filename2)
    save_data_dir = "../scripts/data"
    value_input = sio.loadmat(
        '/home/firefly/chh_ws/src/plan_cdpr/scripts/data/'+Filename1+'.mat')

    point_end_effector = np.array(
        [np.array([0.0, 0.29, 0.25]), np.array([-0.25, -0.145, 0.25]), np.array([0.25, -0.145, 0.25]),
         np.array([0.0, 0.0, 0.25]),
         np.array([0.0, 0.29, -0.25]), np.array([-0.25, -0.145, -0.25]), np.array([0.25, -0.145, -0.25])])

    # 'CABLE_ONE_SIDE', 'CABLE_OTHER_SIDE', 'POSE_0', 'ROTATION_CENTER', 'point_end_effector'
    print(value_input.keys())
    # print(value_input['CABLE_OTHER_SIDE'][0])
    wrench_force = sio.loadmat(
        '/home/firefly/chh_ws/src/plan_cdpr/scripts/data/'+Filename2+'.mat')
    print(wrench_force.keys())
    plot_r_t = []
    plot_r_r = []
    plot_r = []
    r_r_AWs = []
    r_t_AWs = []
    plot_r_all = []
    rs = []
    print(len(wrench_force['wrench']))
    for i in range(len(wrench_force['wrench'])):
        CABLE_ONE_SIDE = value_input['CABLE_ONE_SIDE'][i]
        CABLE_OTHER_SIDE = value_input['CABLE_OTHER_SIDE'][i]
        rotation_cent = value_input['ROTATION_CENTER'][i]
        pose_0 = value_input['POSE_0'][i]
        # print(value_input['CABLE_OTHER_SIDE'][2])
        # print('v', value_input['ROTATION_CENTER'][2])
        # print(np.shape(CABLE_ONE_SIDE))
        # print(np.shape(CABLE_OTHER_SIDE))

        value.set_jacobian_param(point_end_effector, pose_0, rotation_cent)
        # v = value.cost_cable_interference(CABLE_ONE_SIDE, CABLE_OTHER_SIDE)
        # v1 = value.cost_cable_length(CABLE_ONE_SIDE, CABLE_OTHER_SIDE)
        # v2 = value.cost_feasible_points(CABLE_ONE_SIDE, CABLE_OTHER_SIDE, np.array([5, 5, 5, 5, 5, 5, 5]))
        r_t_AW = value.r_t_AW(CABLE_ONE_SIDE, CABLE_OTHER_SIDE)
        r_r_AW = value.r_r_AW(CABLE_ONE_SIDE, CABLE_OTHER_SIDE, rotation_cent)
        
        # print('cost_cable_interference: {}\ncost_cable_length: {}\ncost_feasible_points: {}'.format(
        #     v, v1, v2))
        # print('r_r_AW: {}\nr_t_AW: {}'.format(r_r_AW, r_t_AW))
        rs.append(r_t_AW + r_r_AW)
        r_r_AWs.append(r_r_AW)
        r_t_AWs.append(r_t_AW)
        plot_r_r.append(-r_r_AW/1.0)
        plot_r_t.append(-r_t_AW/10000)
        plot_r.append(-r_r_AW/1.0-r_t_AW/10000)

        wrench = wrench_force['wrench'][i]
        jac, _, _ = jacobi(CABLE_OTHER_SIDE, point_end_effector, pose_0, rotation_cent, wrench)
        v_r = value.cost_r(jac)
        print(v_r)
        plot_r_all.append(-v_r)
    # plot_r_all = smooth(plot_r_all,3)




    # sio.savemat(save_data_dir + '/r_' + Filename1+'.mat',
    #             dict(r_t_AWs=r_t_AWs, r_r_AWs=r_r_AWs, rs=rs))
    plt.subplot(311)
    plt.plot(plot_r_r, 'r')
    plt.legend(labels=['r_r_AW'], loc='best')

    plt.subplot(312)
    plt.plot(plot_r_t, 'g')
    plt.legend(labels=['r_t_AW'], loc='best')
    # plt.savefig('../scripts/figure/r_.png')

    plt.subplot(313)
    plt.plot(plot_r, 'y')
    plt.legend(labels=['r'], loc='best')
    plt.savefig('../scripts/figure/r_.png')

    plt.figure()
    plt.plot(plot_r_all)
    plt.savefig('../scripts/figure/r_all_1.png')

    plt.show()


if __name__ == '__main__':
    # Filenames = ['static_01_value_input_2021-11-10-21:13:10', 'static_01_value_input_2021-11-10-21:14:08', 'static_01_value_input_2021-11-10-21:15:05', 
    # 'static_01_value_input_2021-11-10-21:17:25', 'static_01_value_input_2021-11-10-21:18:23', 'static_01_value_input_2021-11-10-21:19:20',
    # 'static_01_value_input_2021-11-10-21:20:18', 'static_01_value_input_2021-11-10-21:21:15', 'static_01_value_input_2021-11-10-21:22:13', 
    # 'static_01_value_input_2021-11-10-21:23:11', 'static_01_value_input_2021-11-10-21:24:09', 'static_02_value_input_2021-11-10-21:28:41', 
    # 'static_02_value_input_2021-11-10-21:29:39',  'static_02_value_input_2021-11-10-21:30:36',  'static_02_value_input_2021-11-10-21:31:34', 
    # 'static_02_value_input_2021-11-10-21:32:32',  'static_02_value_input_2021-11-10-21:33:30',  'static_02_value_input_2021-11-10-21:38:40', 
    # 'static_02_value_input_2021-11-10-21:39:38',  'static_02_value_input_2021-11-10-21:40:36',  'static_02_value_input_2021-11-10-21:41:33', 
    # 'static_02_value_input_2021-11-10-21:42:31', ]
    
    Filenames = ['static_02_value_input_2021-11-11-21:28:59']
    file_name2 = ['static_02_force_input_2021-11-11-21:28:59']
    for i in range(len(Filenames)):
        calculate_r(Filenames[i], file_name2[i])
