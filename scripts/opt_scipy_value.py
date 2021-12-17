import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares
from distance_between_lines import *
from module_test.null_test import *
from value_opti import *

'''
优化无人机的x，y，共8维
'''


def SQPfun(args):
    cables_one_side, cable_oth_side, m, d_min, mu_l, mu_t, mu_r, mu_d, mu_a, rotation_center, target_length_list, t_min, t_max, point_end_effector, pose_0 = args  # 21
    cables_one_side = np.array(cables_one_side)
    cable_oth_side = np.array(cable_oth_side)
    # print(np.shape(cable_oth_side))  #21
    cables_one_side = np.reshape(cables_one_side, (7, 3))
    # print(cables_one_side)
    t_max = np.reshape(t_max, (7, 1))
    t_min = np.reshape(t_min, (7, 1))

    def fun(x):
        x = np.array(x)
        x_uav = np.insert(x, [2, 4, 6, 8], [8.0,8.0,8.0,8.0])
        cables_other_side = np.concatenate((x_uav, cable_oth_side))
        cables_other_side = np.reshape(cables_other_side, (7, 3))
        # print(np.shape(cables_other_side))  #21
        # cost1 = cost_cable_length(cables_one_side, cables_other_side, mu_l, target_length_list)
        cost2 = cost_cable_interference(
            cables_one_side, cables_other_side, d_min, mu_d)
        # FIXME:RuntimeWarning: overflow encountered in exp
        cost3 = cost_feasible_points(
            cables_other_side, mu_a)
        # # cost3 = 0
        # b = cables_one_side - rotation_center
        # U = cables_other_side - cables_one_side
        # # print('b', np.shape(b))  #7*3
        # # print('U', np.shape(U))  #7*3
        # # print(U)
        # b_x_U = []
        # u = []
        # for i in range(0, len(cables_one_side), 1):
        #     u_norm = U[i] / np.linalg.norm(U[i])
        #     u.append(u_norm)
        #     b_x_U.append(np.cross(b[i], u[i]))
        # # print(np.shape(b_x_U))  # 7*3
        # # print(b_x_U)  # 7*3
        # d1, u_null1 = AW(np.array(u).T, m, t_min, t_max)
        # r_A1 = r_AW(d1, u_null1)
        # d2, u_null2 = AW(np.array(b_x_U).T, m, t_min, t_max)
        # r_A2 = r_AW(d2, u_null2)
        # cost4 = cost_wrench_function([mu_t * r_A1, mu_r * r_A2])
        # r_r = r_r_AW(cables_one_side, cables_other_side, rotation_center, t_min,t_max,mu_r)
        r_t = r_t_AW(cables_one_side, cables_other_side, t_min,t_max,mu_t,m)
        cost4 =  r_t
        # bool_judge = judge_tensions(cables_other_side, point_end_effector,pose_0,rotation_center)
        # if not bool_judge:
        #     cost4 = 0
        return cost4 + cost3 + cost2

    return fun


def con1():
    cons = []
    for i in range(8):
        con = {'type': 'ineq', 'fun': lambda x, i=i: -np.abs(x[i]) + 10}
        cons.append(con)
    return cons


def minimize_cost(allArgs):
    cons = con1()
    cables_one_side, cable_oth_side, m, d_min, mu_l, mu_t, mu_r, mu_d, mu_a, rotation_center, target_length_list, t_min, t_max, uav_pos, point_end_effector,pose_0 = allArgs  # 21
    Args = [cables_one_side, cable_oth_side, m, d_min, mu_l, mu_t, mu_r, mu_d, mu_a,
            rotation_center, target_length_list, t_min, t_max, point_end_effector, pose_0]
    x0 = np.asarray(uav_pos)
    # tol=8也是迭代次数为1 , options={'disp': True, 'maxiter': 1}
    res = minimize(SQPfun(Args), x0, method='SLSQP', constraints=cons, options={'disp': True}, tol=8)
    v = res.x  # 返回最小化后的变量
    result = res.fun  # 返回最小化后的结果
    print('*******************')
    print('cost_result', result)
    print(res.x)
    print()
    print(res.success)
    print()
    return v


if __name__ == "__main__":
    import time

    s = time.time()
    d1 = np.linalg.norm([0, -3, 4.5])
    d2 = np.linalg.norm([5, 5, 3])
    cable_one_side = np.array([[0, 0, 4.25],
                                   [-0.25, -0.145, 4.25],
                                   [0.25, -0.145, 4.25],
                                   [0, 0, 4.25],
                                   [0, 0, 3.75],
                                   [-0.25, -0.145, 3.75],
                                   [0.25, -0.145, 3.75]])
    cable_one_side =np.array( [[0.0,0.0,4.0] for _ in range(7)])
    print(cable_one_side)
    x_ugv = np.array([2.5, 1.45, 0, -2.5, 1.45, 0, 0, -2.9, 0])  # 9
    point_end_effector = np.array(
        [np.array([0.0, 0, 0.5]), np.array([-0, -0, 0.5]), np.array([0, -0.5, 0]),
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 0.5, -0]), np.array([-0, -0.5, -0]), np.array([0, -0.5, 0.0])])
    uav_pos = np.array([-2.5, 1.45,0, -2.9,2.5, 1.45,0, 0])
    pose_0 = [0, 0, 0, 1]
    rotation_center = [0, 0, 3]
    args = (cable_one_side,
            x_ugv,
            0.1, 0.1, 0.1, 1.0, 1.0, 0.2, 0.2,
            rotation_center,
            [1, 2, 1, 1, 1, 1, 1],
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            np.array([5, 5, 5, 5, 5, 5, 5]),
            uav_pos,
            point_end_effector,
            pose_0)
    v = minimize_cost(args)
    print(time.time() - s)
