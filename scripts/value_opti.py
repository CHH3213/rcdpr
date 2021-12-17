# -*-coding:utf-8-*-

from distance_between_lines import *
from module_test.null_test import *
import numpy as np
# import jax.numpy as np
import math
from Jacobian import JacobianAndForce

'''
定义代价函数类：包含4种代价函数,用于优化
'''



def cost_cable_length(cables_one_side, cables_other_side,mu_l,target_length_list):
    '''
    length of cable cost
    :param cables_one_side: 当前平台固定点位置集合（7个）维度：7*3
    :param cables_other_side: 变量；当前agent固定点位置，也即agent自身位置（7个），维度：7*3
    cable_length_list: the length of all cables
    :return: cost
    '''
    sum_list = []
    cable_lines = np.array(
        [np.array([cables_one_side[i]-cables_other_side[i]]) for i in range(len(cables_one_side))])
    cable_length_list = [np.linalg.norm(cable_lines[i]) for i in range(len(cable_lines))]
    for i, cable_length_i in enumerate(cable_length_list):
        cable_length_i = np.array(cable_length_i)
        f_iL = mu_l * np.linalg.norm(cable_length_i - target_length_list[i])
        sum_list.append(f_iL)
    return np.sum(np.array(sum_list))

def cost_cable_interference(cables_one_side, cables_other_side,d_min,mu_d):
    '''
    绳子碰撞代价
    :param cables_one_side: 当前平台固定点位置集合（7个）维度：7*3
    :param cables_other_side: 变量；当前agent固定点位置，也即agent自身位置（7个），维度：7*3
    lines:线段数组，np.array([line1([array1,array2]),line1([array1,array2]),...])一个array为1个端点，两个array组成一条线段, 末端点即为变量
    :return: 返回碰撞代价
    '''
    # lines = np.array([np.array([cables_one_side[i], cables_other_side[i]]) for i in range(len(cables_one_side))])
    # # print('lines shape', np.shape(lines))
    # assert len(lines) >= 2, "至少需要两条线段，才可求距离"
    # sum_list = []
    # for i in range(len(lines)):
    #     assert len(lines[i]) % 2 == 0, "线段是由两个端点描述，请检查端点数是否正确！"
    #     for j in range(i + 1, len(lines)):
    #         _, _, d_ij = closestDistanceBetweenLines(lines[i, 0], lines[i, 1], lines[j, 0], lines[j, 1])
    #         # print('d_ij',d_ij)
    #         f_ij = np.exp(-(d_ij - d_min) / mu_d)
    #         sum_list.append(f_ij)
    # sum_f = np.sum(np.array(sum_list))
    # return sum_f

    lines = np.array([np.array([cables_one_side[i], cables_other_side[i]])
                      for i in range(len(cables_one_side))])
    # print('lines shape', np.shape(lines))
    assert len(lines) >= 2, "至少需要两条线段，才可求距离"
    sum_list = []

    rtod = 180/np.pi
    for i in range(len(lines)):
        assert len(lines[i]) % 2 == 0, "线段是由两个端点描述，请检查端点数是否正确！"

        # TODO: 计算两条线段的夹角
        line1 = lines[i, 1] - lines[i, 0]
        for j in range(i + 1, len(lines)):
            line2 = lines[j, 1] - lines[j, 0]
            d_ij = math.acos(np.einsum('i,i->', line1, line2) /
                                (np.linalg.norm(line1) * np.linalg.norm(line2)))

            f_ij = np.exp(-(d_ij * rtod - d_min) / mu_d)
            # print(d_ij * rtod)
            sum_list.append(f_ij)
    sum_f = np.sum(sum_list)
    # print(sum_f)
    return sum_f

def cost_feasible_points(cables_other_side, mu_a,cables_one_side=None, feasible_sets=None, bound_set=None):
    """
    可行的固定点位置代价——绳长约束
    :param cables_one_side: 当前平台固定点位置集合（7个）维度：7*3
    :param cables_other_side: 变量；当前agent固定点位置，也即agent自身位置（7个），维度：7*3
    anchor_points:全部固定点集合，形式为np.array([[a0,a1],[a2,a3]...,[an-1,an]]),ai是array数组,表示第i个固定点,[a0,a1]
    前者表示是平台上的固定点例如a0，后者是agent上的固定点例如a1,其实就是lines.
    feasible_sets:对应的固定点的可行域描述：以平台固定点为圆心，最长绳长为半径的半球区域（实心）。形式为：[绳长].
    bound_set:对应的固定点的边界域描述：以平台固定点为圆心，最长绳长为半径的半球面。形式为：点集.
    return: 返回可行的位置代价.
    """
    # anchor_points = np.array(
    #     [np.array([cables_one_side[i], cables_other_side[i]]) for i in range(len(cables_one_side)-3)])
    # assert len(anchor_points) == len(feasible_sets) and len(anchor_points) == len(bound_set), "确保每组固定点都有对应的集合"
    # assert len(anchor_points) == len(feasible_sets) and len(
    #     anchor_points), "确保每组固定点都有对应的集合"
    sum_list = []
    center_point = np.array(
        [[-2.5, 2.5], [-2.5, -2.5], [2.5, -2.5], [2.5, 2.5]])
    for i in range(np.shape(center_point)[0]):
        # if np.linalg.norm(anchor_points[i, 0] - anchor_points[i, 1]) > feasible_sets[i] or anchor_points[i, 0, 2] >= \
        #         anchor_points[i, 1, 2]:
        # # f_i = np.exp(-np.min([anchor_points[i, 1]-bound_set[j] for j in range(len(bound_set))])/self.mu_a)
        #     f_i = np.exp((np.linalg.norm(anchor_points[i, 1] - anchor_points[i, 0]) - feasible_sets[i]) / self.mu_a)
        # else:
        #     f_i = np.exp(-(np.linalg.norm(anchor_points[i, 1] - anchor_points[i, 0]) - feasible_sets[i]) / self.mu_a)
        # f_i = np.exp((np.linalg.norm(
        #     anchor_points[i, 1] - anchor_points[i, 0]) - feasible_sets[i]) / self.mu_a)
        f_i = np.exp((np.linalg.norm(
            cables_other_side[i, :2] - center_point[i, :]) - 2.5) / mu_a)
        # print(cables_other_side[i, :2])
        # print(f_i)
        # if f_i > 1000:
        #     print('point_1:{}\npoint_2:{}\nfeasible:{}\nf_i:{}\n'.format(
        # anchor_points[i, 1], anchor_points[i, 0], feasible_sets[i], f_i))
        # print('anchor_points[i, 1], anchor_points[i, 0], f_i', anchor_points[i, 1]-anchor_points[i, 0], f_i)
        sum_list.append(f_i)
    # print('sum_list', sum_list)
    return np.sum(sum_list)

def AW(U,m, t_min,t_max, name='U' ):
    """
    计算AW
    :param name: choose U or bxu
    :param U:雅可比矩阵中的U或bXu  3*7
    :return: d, u_null
    """
    M = []
    u_null = []
    C = []
    d = []
    I_plus = []
    I_minus = []
    # print('U.T',np.shape(U.T))
    # print('len:U.T',len(U.T))
    # print(U.T[1])t_max
    '''计算0空间'''
    for i in range(len(U.T)):
        for j in range(i + 1, len(U.T)):
            # parallel = is_parallel(U.T[i], U.T[j])
            parallel = np.cross(U.T[i], U.T[j])

            # print(parallel)
            if np.linalg.norm(parallel):
            # if not parallel:
                M_i = [U.T[i], U.T[j]]  # 2*3
                # print(np.shape(M_i))
                # print('M_i',M_i)
                u_null_i = null_space(M_i)  # 3*1
                # print('u_null_i', u_null_i)
                # print(np.shape(u_null_i))
                c_i = [u_null_i, -u_null_i]  # 2*3
                # print(np.shape(c_i))
                # print('shape',np.shape(U.T[j].T))
                M.append(M_i)
                u_null.append(u_null_i)
                C.append(c_i)
            else:
                continue
    '''求I_plus和I_minus'''
    for u_null_i in u_null:
        for j in range(len(U.T)):
            # print('dot', np.dot(u_null_i.T, U.T[j].T))
            # print('*', np.sum(u_null_i.T*U.T[j].T))
            if np.any(U.T) == np.nan:
                break
            if np.sum(np.dot(u_null_i.T, U.T[j].T)) > 1e-10:
                I_plus.append(j)  # save index
            else:
                I_minus.append(j)
    for i in range(len(u_null)):
        if name == 'U':
            # print(np.shape(t_max))
            # print(np.dot(t_max, u_null[i].T))
            # print(np.dot(u_null[i].T,np.array([0, 0, -9.8])))
            d_pi = np.sum([np.dot(np.dot(t_max, u_null[i].T), U.T[j].T) for j in I_plus]) \
                   + np.sum([np.dot(np.dot(t_min, u_null[i].T), U.T[j].T) for j in I_minus]) \
                   - m * np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
            d_qi = -np.sum([np.dot(np.dot(t_max, u_null[i].T), U.T[j].T) for j in I_minus]) \
                   - np.sum([np.dot(np.dot(t_min, u_null[i].T), U.T[j].T) for j in I_plus]) \
                   - m * np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
        else:
            d_pi = np.sum([np.dot(np.dot(t_max, u_null[i].T), U.T[j].T) for j in I_plus]) \
                   + np.sum([np.dot(np.dot(t_min, u_null[i].T), U.T[j].T) for j in I_minus])
            d_qi = -np.sum([np.dot(np.dot(t_max, u_null[i].T), U.T[j].T) for j in I_minus]) \
                   - np.sum([np.dot(np.dot(t_min, u_null[i].T), U.T[j].T) for j in I_plus])
        d.append([d_pi, d_qi])
    return d, u_null

def r_AW(d, u_null):
    """
    计算r_AW
    :param d: d_pi,d_qi的集合  列表，形式：[[],[]]
    :param u_null: u_null_i的集合
    :return: r_AW的值
    """
    assert len(u_null) == len(d), "数组长度需要保持一致"
    min_in_list = []
    for i in range(len(u_null)):
        min_in = np.min([np.linalg.norm(d[i][0]) / np.linalg.norm(u_null[i]),
                         np.linalg.norm(d[i][1]) / np.linalg.norm(u_null[i])])
        min_in_list.append(min_in)
    return np.min(min_in_list)

def cost_wrench_function(r_AW):
    """
    可行的力空间
    :param r_AW: [r_t_AW, r_r_AW]
    :return: cost
    """
    return -np.sum(r_AW)

def r_r_AW(cables_one_side, cables_other_side, rotation_center, t_min,t_max,mu_r):
    """

    :param cables_one_side:
    :param cables_other_side:
    :param rotation_center: 末端执行器坐标系相对于基坐标系的位置
    :return:
    """
    # print(' shape-cables_one_side', np.shape(cables_one_side))
    b = cables_one_side - rotation_center
    U = cables_other_side - cables_one_side
    # print('b', np.shape(b))
    b_x_U = []
    u = []
    for i in range(0, len(cables_one_side), 1):
        u_norm = U[i] / np.linalg.norm(U[i])
        u.append(u_norm)
        b_x_U.append(np.cross(b[i], u[i]))
    U = np.array(b_x_U).T
    M = []
    u_null = []
    C = []
    d = []
    I_plus = []
    I_minus = []
    # print('U.T',np.shape(U.T))
    # print('len:U.T',len(U.T))
    # print(U.T[1])
    '''计算0空间'''
    for i in range(len(U.T)):
        for j in range(i + 1, len(U.T)):
            # parallel = is_parallel(U.T[i], U.T[j])
            parallel = np.cross(U.T[i], U.T[j])

            # print(parallel)
            if np.linalg.norm(parallel):
            # if not parallel:
                M_i = [U.T[i], U.T[j]]  # 2*3
                # print(np.shape(M_i))
                u_null_i = null_space(M_i)  # 3*1
                # print(u_null_i)
                # print(np.shape(u_null_i))
                c_i = [u_null_i, -u_null_i]  # 2*3
                # print(np.shape(c_i))
                # print('shape',np.shape(U.T[j].T))
                M.append(M_i)
                u_null.append(u_null_i)
                C.append(c_i)
            else:
                continue
    '''求I_plus和I_minus'''
    for u_null_i in u_null:
        for j in range(len(U.T)):
            # print('dot', np.dot(u_null_i.T, U.T[j].T))
            # print('*', np.sum(u_null_i.T*U.T[j].T))
            if np.sum(np.dot(u_null_i.T, U.T[j].T)) > 1e-10:
                I_plus.append(j)  # save index
            else:
                I_minus.append(j)
    for i in range(len(u_null)):
        # print(np.shape(t_max))
        print(np.shape(u_null[i]))
        # print(np.dot(t_max, u_null[i].T))
        # print(np.dot(u_null[i].T,np.array([0, 0, -9.8])))
        d_pi = np.sum([np.dot(np.dot(t_max, u_null[i].T), U.T[j].T) for j in I_plus]) \
               + np.sum([np.dot(np.dot(t_min, u_null[i].T), U.T[j].T) for j in I_minus])
        d_qi = -np.sum([np.dot(np.dot(t_max, u_null[i].T), U.T[j].T) for j in I_minus]) \
               - np.sum([np.dot(np.dot(t_min, u_null[i].T), U.T[j].T) for j in I_plus])
        d.append([d_pi, d_qi])

    """计算r_AW"""
    assert len(u_null) == len(d), "数组长度需要保持一致"
    min_in_list = []
    for i in range(len(u_null)):
        min_in = np.min([np.linalg.norm(d[i][0]) / np.linalg.norm(u_null[i]),
                         np.linalg.norm(d[i][1]) / np.linalg.norm(u_null[i])])
        min_in_list.append(min_in)
    return -mu_r*np.min(min_in_list)

def r_t_AW(cables_one_side, cables_other_side, t_min,t_max,mu_t, m):
    """

        :param cables_one_side:
        :param cables_other_side:
        :return:
        """
    U = cables_other_side - cables_one_side
    # print(U)
    # print(np.shape(U))
    # print('len(cables_one_side)', len(cables_one_side))
    u = []
    for i in range(0, len(cables_one_side), 1):
        u_norm = U[i] / np.linalg.norm(U[i])
        u.append(u_norm)
        # print(u_norm)
    U = np.array(u).T
    # print(np.shape(U))
    M = []
    u_null = []
    C = []
    d = []
    I_plus = []
    I_minus = []
        # print('U.T',np.shape(U.T))
        # print('len:U.T',len(U.T))
        # print(U.T[1])
    '''计算0空间'''
    for i in range(len(U.T)):
        for j in range(i + 1, len(U.T)):
            # parallel = is_parallel(U.T[i], U.T[j])
            parallel = np.cross(U.T[i], U.T[j])
            # print('parallel', parallel)
            if np.linalg.norm(parallel) > 1e-10:
                # if not parallel:
                M_i = [U.T[i], U.T[j]]  # 2*3
                # print('M_i', M_i)
                # print('M_i shape', np.shape(M_i))
                u_null_i = nullspace(M_i)  # 3*1
                # print('u_null_i', u_null_i)
                # print('u_null_i shape', np.shape(u_null_i))
                c_i = [u_null_i, -u_null_i]  # 2*(3*1)
                # print(c_i)
                # print('shape',np.shape(U.T[j].T))
                M.append(M_i)
                u_null.append(u_null_i)
                C.append(c_i)
            else:
                continue
    '''求I_plus和I_minus'''
    I_pluss = []
    I_minuss = []
    for u_null_i in u_null:
        I_plus = []
        I_minus = []
        for j in range(len(U.T)):
            # print('dot', np.dot(u_null_i.T, U.T[j].T))
            # print('*', np.sum(u_null_i.T*U.T[j].T))
            if np.sum(np.dot(u_null_i.T, U.T[j].T)) > 1e-10:
                I_plus.append(j)  # save index
            else:
                I_minus.append(j)

        I_pluss.append(I_plus)
        I_minuss.append(I_minus)

    for i in range(len(u_null)):
        # print('t_max', np.shape(t_max))
        # print(np.dot(t_max, u_null[i].T))
        # print(np.dot(u_null[i].T,np.array([0, 0, -9.8])))
        # print(np.einsum('i,i->', u_null[i].T[0], np.array([0, 0, -9.8])))
        # print(u_null[i])
        # d_pi = np.sum([np.dot(np.dot(t_max, u_null[i].T), U.T[j].T) for j in I_pluss[i]]) \
        #     + np.sum([np.dot(np.dot(t_min, u_null[i].T), U.T[j].T) for j in I_minuss[i]]) \
        #     - m * \
        #     np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
        # d_qi = -np.sum([np.dot(np.dot(t_max, u_null[i].T), U.T[j].T) for j in I_minuss[i]]) \
        #        - np.sum([np.dot(np.dot(t_min, u_null[i].T), U.T[j].T) for j in I_pluss[i]]) \
        #        - m * \
        #     np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
        d_pi = 5.0*np.sum([np.dot(u_null[i].T, U.T[j].T) for j in I_pluss[i]]) \
            + 0.1 * np.sum([np.dot(u_null[i].T, U.T[j].T) for j in I_minuss[i]]) \
            - m * \
            np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
        d_qi = -5.0*np.sum([np.dot(u_null[i].T, U.T[j].T) for j in I_minuss[i]]) \
                - 0.1 * np.sum([np.dot(u_null[i].T, U.T[j].T) for j in I_pluss[i]]) \
                - m * \
            np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
        # print('d_pi', np.shape(np.dot(np.dot(t_max, u_null[0].T), U.T[0].T)))
        d.append([d_pi, d_qi])
        # if d_pi > 35 or d_qi < -35:
        #     # print(d_pi, d_qi)

        #     print('sum: ', np.sum([np.dot(t_max,np.dot( u_null[i].T, U.T[j].T)) for j in I_minuss[i]]))
        #     print('a',5.0 * np.sum([np.dot(u_null[i].T[0], U.T[j].T) for j in I_minuss[i]]) )
        #     print(t_max.T, u_null[i].T, U.T[j].T)

    """计算r_AW"""
    assert len(u_null) == len(d), "数组长度需要保持一致"
    min_in_list = []
    for i in range(len(u_null)):
        # print(np.linalg.norm(u_null[i]))
        min_in = np.min([np.linalg.norm(d[i][0]) / np.linalg.norm(u_null[i]),
                            np.linalg.norm(d[i][1]) / np.linalg.norm(u_null[i])])
        min_in_list.append(min_in)

    return -mu_t * np.min(min_in_list)

def judge_tensions(points_base, point_end_effector, pose_0, rotation_center):
    '''
    判断是否有工作空间
    :param points_base:agent 固定点：即cable_other_side
    :param point_end_effector: 末端的位置，固定
    :param pose_0: platform姿态
    :param rotation_center:platform 位置
    :return:bool, 有小于0的拉力，返回False，否则返回True
    '''
    jac = JacobianAndForce()
    J = jac.get_jacobian(
        points_base, point_end_effector, pose_0, rotation_center)
    j_rotation = J
    # print cable_force_vector
    # qiu通解  这边：J=[J1(6*6),J2(6*1)]：6*7， t=[t1(6*1) t2(1*1)]：7*1， w：6*1  J1*t1+J2*t2 = 0,默认t2=1
    cable_others = j_rotation[:, 0:6]  # 6*6矩阵
    cable_one = -j_rotation[:, 6]   # 6*1矩阵
    # equivalence = (cable_others)^(-1)*cable_one
    equivalence = np.linalg.solve(cable_others, cable_one)
    for i in range(0, 6, 1):
        if equivalence[i] < 0:
            return False
        # print('e', equivalence[i])
    return True 
