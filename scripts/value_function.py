# -*-coding:utf-8-*-

from distance_between_lines import *
from module_test.null_test import *
import numpy as np
import math
import scipy.io as sio
from Jacobian import JacobianAndForce
import matplotlib.pyplot as plt

'''
定义代价函数类：包含4种代价函数
'''


class Value_function:
    def __init__(self, m=0.1, d_min=5.0, mu_t=3.0, mu_r=1.0, mu_d=0.02, mu_a=0.03,
                 t_min=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                 t_max=np.array([5, 5, 5, 5, 5, 5, 5]), mu_l=0.2,
                 target_length_list=np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])):
        """
        :param mu_t: penality coefficient of r_t_AW
        :param mu_r:  penality coefficient of r_r_AW
        :param m: mass of platform
        :param d_min: minimum distance between two cables
        :param mu_d: cable distance penality coefficient
        :param mu_a:  an anchor point penality coefficient
        :param t_min:  the vector of minimum tension
        :param t_max:  the vector of maximum tension
        :param mu_l:  cable length penality coefficient
        :param target_length_list: target length of the ith cable
        """
        # TODO：系数的确定
        self.mu_r = mu_r
        self.mu_t = mu_t
        self.m = m
        self.d_min = d_min  # scalar
        self.mu_d = mu_d  # scalar
        self.mu_a = mu_a  # scalar
        self.t_min = t_min  # 7*1
        self.t_max = t_max  # 7*1
        self.t_max = np.reshape(self.t_max, (7, 1))
        self.t_min = np.reshape(self.t_min, (7, 1))
        self.mu_l = mu_l  # scalar
        self.target_length_list = target_length_list  # 7*1
        self.Jac = JacobianAndForce()

    def save_cfg(self):
        sio.savemat('data/value_func_cfg.mat', dict(mu_r = self.mu_r, mu_t = self.mu_t, m = self.m, d_min = self.d_min, mu_d = self.mu_d, mu_a = self.mu_a,
        t_min = self.t_min, t_max = self.t_max, mu_l = self.mu_l, target_length_list = self.target_length_list))

    def set_jacobian_param(self, point_end_effector, pose_0, rotation_center):
        '''
        设置雅可比参数
        :param point_end_effector:
        :param pose_0:
        :param rotation_center:
        :return:
        '''
        self.point_end_effector = np.array(point_end_effector)
        self.pose_0 = np.array(pose_0)
        self.rotation_center = np.array(rotation_center)

    def AW_part(self, U, name='U'):
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
        # print(U.T[1])
        '''计算0空间'''
        for i in range(len(U.T)):
            for j in range(i + 1, len(U.T)):
                # parallel = is_parallel(U.T[i], U.T[j])
                parallel = np.cross(U.T[i], U.T[j])
                # print(parallel)
                # if not parallel:
                if np.linalg.norm(parallel):
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
                if np.dot(u_null_i.T, U.T[j].T) > 1e-10:
                    I_plus.append(j)  # save index
                else:
                    I_minus.append(j)
        for i in range(len(u_null)):
            if name == 'U':
                # print(np.shape(self.t_max))
                # print(np.dot(self.t_max, u_null[i].T))
                # print(np.dot(u_null[i].T,np.array([0, 0, -9.8])))
                d_pi = np.sum([np.dot(np.dot(self.t_max, u_null[i].T), U.T[j].T) for j in I_plus]) \
                    + np.sum([np.dot(np.dot(self.t_min, u_null[i].T), U.T[j].T) for j in I_minus]) \
                    - self.m * \
                    np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
                d_qi = -np.sum([np.dot(np.dot(self.t_max, u_null[i].T), U.T[j].T) for j in I_minus]) \
                       - np.sum([np.dot(np.dot(self.t_min, u_null[i].T), U.T[j].T) for j in I_plus]) \
                       - self.m * \
                    np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
            else:
                d_pi = np.sum([np.dot(np.dot(self.t_max, u_null[i].T), U.T[j].T) for j in I_plus]) \
                    + np.sum([np.dot(np.dot(self.t_min, u_null[i].T), U.T[j].T)
                             for j in I_minus])
                d_qi = -np.sum([np.dot(np.dot(self.t_max, u_null[i].T), U.T[j].T) for j in I_minus]) \
                       - np.sum([np.dot(np.dot(self.t_min, u_null[i].T), U.T[j].T)
                                for j in I_plus])
            d.append([d_pi, d_qi])
        return d, u_null

    def r_AW(self, d, u_null):
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

    def cost_wrench_function(self, r_AW):
        """
        可行的力空间
        :param r_AW: [r_t_AW, r_r_AW]
        :return: cost
        """
        return -np.sum(r_AW)

    def cost_cable_length(self, cables_one_side, cables_other_side):
        '''
        length of cable cost
        :param cables_one_side: 当前平台固定点位置集合（7个）维度：7*3
        :param cables_other_side: 变量；当前agent固定点位置，也即agent自身位置（7个），维度：7*3
        cable_length_list: the length of all cables
        :return: cost
        '''
        sum_list = []
        cable_lines = np.array(
            [np.array([cables_one_side[i] - cables_other_side[i]]) for i in range(len(cables_one_side))])
        cable_length_list = [np.linalg.norm(
            cable_lines[i]) for i in range(len(cable_lines))]
        for i, cable_length_i in enumerate(cable_length_list):
            cable_length_i = np.array(cable_length_i)
            f_iL = self.mu_l * \
                np.linalg.norm(cable_length_i - self.target_length_list[i])
            sum_list.append(f_iL)
        return np.sum(sum_list)

    # FIXME:绳子碰撞
    def cost_cable_interference(self, cables_one_side, cables_other_side):
        '''
        绳子碰撞代价
        :param cables_one_side: 当前平台固定点位置集合（7个）维度：7*3
        :param cables_other_side: 变量；当前agent固定点位置，也即agent自身位置（7个），维度：7*3
        lines:线段数组，np.array([line1([array1,array2]),line1([array1,array2]),...])一个array为1个端点，两个array组成一条线段, 末端点即为变量
        :return: 返回碰撞代价
        '''
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

                f_ij = np.exp(-(d_ij * rtod - self.d_min) / self.mu_d)
                # print(d_ij * rtod)
                sum_list.append(f_ij)
        sum_f = np.sum(sum_list)
        # print(sum_f)
        return sum_f

    def cost_feasible_points(self, cables_other_side, cables_one_side=None , feasible_sets=None, bound_set=None):
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
        center_point = np.array([[-2.5,2.5],[-2.5,-2.5],[2.5,-2.5],[2.5,2.5]])
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
                cables_other_side[i, :2] - center_point[i, :]) - 2.5) / self.mu_a)
            # print('cables_other_side[i, :2]', cables_other_side[i, :2])
            # print((np.linalg.norm(cables_other_side[i, :2] - center_point[i, :]) - 2.5))
            # print(f_i)
            # if f_i > 1000:
            #     print('point_1:{}\npoint_2:{}\nfeasible:{}\nf_i:{}\n'.format(
                    # anchor_points[i, 1], anchor_points[i, 0], feasible_sets[i], f_i))
            # print('anchor_points[i, 1], anchor_points[i, 0], f_i', anchor_points[i, 1]-anchor_points[i, 0], f_i)
            sum_list.append(f_i)
        # print('sum_list', sum_list)
        return np.sum(sum_list)

    def r_r_AW(self, cables_one_side, cables_other_side, rotation_center):
        """
        :param cables_one_side:
        :param cables_other_side:
        :param rotation_center: platform坐标系相对于基坐标系的位置
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
            # print(np.shape(self.t_max))
            # print(np.dot(self.t_max, u_null[i].T))
            # print(np.dot(u_null[i].T,np.array([0, 0, -9.8])))
            d_pi = np.sum([np.dot(np.dot(self.t_max, u_null[i].T), U.T[j].T) for j in I_plus]) \
                + np.sum([np.dot(np.dot(self.t_min, u_null[i].T), U.T[j].T)
                         for j in I_minus])
            d_qi = -np.sum([np.dot(np.dot(self.t_max, u_null[i].T), U.T[j].T) for j in I_minus]) \
                   - np.sum([np.dot(np.dot(self.t_min, u_null[i].T), U.T[j].T)
                            for j in I_plus])
            d.append([d_pi, d_qi])
        # print(d)
        """计算r_AW"""
        assert len(u_null) == len(d), "数组长度需要保持一致"
        min_in_list = []
        for i in range(len(u_null)):
            min_in = np.min([np.linalg.norm(d[i][0]) / np.linalg.norm(u_null[i]),
                             np.linalg.norm(d[i][1]) / np.linalg.norm(u_null[i])])
            min_in_list.append(min_in)
            # print(min_in)
        judge = self.judge_tensions(
            cables_other_side, self.point_end_effector, self.pose_0, self.rotation_center)
        if not judge:
            return 0
        else:
            return -self.mu_r * np.min(min_in_list)

    def r_t_AW(self, cables_one_side, cables_other_side):
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
                if np.linalg.norm(parallel)>1e-10:
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
            # print('t_max', np.shape(self.t_max))
            # print(np.dot(self.t_max, u_null[i].T))
            # print(np.dot(u_null[i].T,np.array([0, 0, -9.8])))
            # print(np.einsum('i,i->', u_null[i].T[0], np.array([0, 0, -9.8])))
            # print(u_null[i])
            # d_pi = np.sum([np.dot(np.dot(self.t_max, u_null[i].T), U.T[j].T) for j in I_pluss[i]]) \
            #     + np.sum([np.dot(np.dot(self.t_min, u_null[i].T), U.T[j].T) for j in I_minuss[i]]) \
            #     - self.m * \
            #     np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
            # d_qi = -np.sum([np.dot(np.dot(self.t_max, u_null[i].T), U.T[j].T) for j in I_minuss[i]]) \
            #        - np.sum([np.dot(np.dot(self.t_min, u_null[i].T), U.T[j].T) for j in I_pluss[i]]) \
            #        - self.m * \
            #     np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
            d_pi = 5.0*np.sum([np.dot( u_null[i].T, U.T[j].T) for j in I_pluss[i]]) \
                +0.1* np.sum([np.dot(u_null[i].T, U.T[j].T) for j in I_minuss[i]]) \
                - self.m * \
                np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
            d_qi = -5.0*np.sum([np.dot( u_null[i].T, U.T[j].T) for j in I_minuss[i]]) \
                   -0.1* np.sum([np.dot( u_null[i].T, U.T[j].T) for j in I_pluss[i]]) \
                   - self.m * \
                np.linalg.norm(np.dot(u_null[i].T, np.array([0, 0, -9.8])))
            # print('d_pi', np.shape(np.dot(np.dot(self.t_max, u_null[0].T), U.T[0].T)))
            d.append([d_pi, d_qi])
            # if d_pi > 35 or d_qi < -35:
            #     # print(d_pi, d_qi)

            #     print('sum: ', np.sum([np.dot(self.t_max,np.dot( u_null[i].T, U.T[j].T)) for j in I_minuss[i]]))
            #     print('a',5.0 * np.sum([np.dot(u_null[i].T[0], U.T[j].T) for j in I_minuss[i]]) )
            #     print(self.t_max.T, u_null[i].T, U.T[j].T)

        """计算r_AW"""
        assert len(u_null) == len(d), "数组长度需要保持一致"
        min_in_list = []
        for i in range(len(u_null)):
            # print(np.linalg.norm(u_null[i]))
            min_in = np.min([np.linalg.norm(d[i][0]) / np.linalg.norm(u_null[i]),
                             np.linalg.norm(d[i][1]) / np.linalg.norm(u_null[i])])
            min_in_list.append(min_in)

        return -self.mu_t * np.min(min_in_list)
            # print(min_in)
        # judge = self.judge_tensions(
        #     cables_other_side, self.point_end_effector, self.pose_0, self.rotation_center)
        # if not judge:
        #     return 0
        # else:
        #     return -self.mu_t * np.min(min_in_list)

    def judge_tensions(self, points_base, point_end_effector, pose_0, rotation_center):
        '''
        判断是否有工作空间
        :param points_base:agent 固定点：即cable_other_side
        :param point_end_effector: 末端的位置，固定
        :param pose_0: platform姿态
        :param rotation_center:platform 位置
        :return:bool, 有小于0的拉力，返回False，否则返回True
        '''
        # FIXME: 需要改维数
        J = self.Jac.get_jacobian(
            points_base, point_end_effector, pose_0, rotation_center)
        j_rotation = J[0:3,:]
        print(np.shape(J))
        # print cable_force_vector
        # qiu通解  这边：J=[J1(6*6),J2(6*1)]：6*7， t=[t1(6*1) t2(1*1)]：7*1， w：6*1  J1*t1+J2*t2 = 0,默认t2=1
        cable_others = j_rotation[:, 0:6]  # 6*6矩阵 FIXME: 3x6矩阵
        cable_one = -j_rotation[:, 6]   # 6*1矩阵FIXME: 3x1矩阵
        # equivalence = (cable_others)^(-1)*cable_one
        equivalence = np.linalg.solve(cable_others, cable_one)
        for i in range(6):
            if equivalence[i] < 0:
                return False
            # print('e', equivalence[i])
        return True

    def cost_r(self, jac):
        M = []
        w_null = []
        C = []
        # jac:6*7
        # count = 1
        # print(np.shape(jac))  # (6, 7)
        for i in range(len(jac.T)):
            for j in range(i + 1, len(jac.T)):
                parallel = is_parallel(jac.T[i], jac.T[j])
                # print(parallel)
                if not parallel:
                    M_i_T = np.delete(jac, [i, j], axis=1)  # 6*5
                    # print(np.shape(M_i_T))
                    M_i = M_i_T.T  # 5*6
                    # print(np.shape(M_i))
                    w_null_i = nullspace(M_i)  # 6*1
                    # print(np.dot(M_i,w_null_i))
                    # print('w_null_i',w_null_i)
                    # print(np.shape(w_null_i))
                    # c_i = [w_null_i, -w_null_i]  # (2,6,1)
                    # print(np.shape(c_i))
                    M.append(M_i)
                    w_null.append(w_null_i)
                    # print(w_null_i)
                    C.append(w_null_i)
                    C.append(-w_null_i)
                    # print(count)  # 21次
                    # count = count + 1
                else:
                    print('wrong, parallel exists!!!')
                    return None
                    # continue
        # print(np.shape(C))  # (42, 6, 1)
        # print(np.shape(w_null))  # (21, 6, 1)
        # print(C[0])
        # print(np.shape(C[0].T))  # (1, 6)

        '''求I_plus和I_minus'''
        I_plus = []
        I_minus = []
        for j, c_j in enumerate(w_null):
            for i, w in enumerate(jac.T):
                # print(w)
                # print(c_j)
                # print(np.dot(c_j.T, w))
                if np.sum(np.dot(c_j.T, w)) > 1e-10:
                    I_plus.append(i)  # save index
                else:
                    I_minus.append(i)
        # print(I_plus)

        '''compute d_j'''
        # print('v',np.shape(jac.T[0]))
        # print('c:{}\t jac:{}\t sum:{}'.format(C[0].T, jac.T[0], np.einsum('ij,j', C[0].T, jac.T[0])))
        # print('v',np.shape(np.array([0, 0, -9.8,0,0,0])))
        # print('fgasd',self.m * np.linalg.norm(np.dot(w_null[1].T, np.array([0, 0, -9.8,0,0,0]))))
        d = []
        for j, c_j in enumerate(w_null):
            # print('t_max', np.shape(self.t_max))
            d_pi = np.sum([self.t_max[0] * np.einsum('ij,j', c_j.T, jac.T[i].T) for i in I_plus]) \
                + np.sum([self.t_min[0] * np.einsum('ij,j', c_j.T, jac.T[i].T) for i in I_minus]) \
                - self.m * \
                np.linalg.norm(
                    np.einsum('ij,j', c_j.T, np.array([0, 0, -9.8, 0, 0, 0])))
            d_qi = -np.sum([self.t_max[0] * np.einsum('ij,j', c_j.T, jac.T[i].T) for i in I_minus]) \
                   - np.sum([self.t_min[0] * np.einsum('ij,j', c_j.T, jac.T[i].T) for i in I_plus]) \
                   - self.m * \
                np.linalg.norm(np.dot(c_j.T, np.array([0, 0, -9.8, 0, 0, 0])))
            d.append([d_pi, d_qi])
        # print(np.shape(d))  # (21, 2)
        """计算r_AW"""
        assert len(w_null) == len(d), "数组长度需要保持一致"
        min_in_list = []
        for i in range(len(d)):
            min_in = np.min([np.linalg.norm(d[i][0]) / np.linalg.norm(w_null[i]),
                             np.linalg.norm(d[i][1]) / np.linalg.norm(w_null[i])])
            min_in_list.append(min_in)

        # 论文
        # d_ = []
        # for j, c_j in enumerate(C):
        #     d_j = np.sum([self.t_max[0]*np.einsum('ij,j',c_j.T,jac.T[i].T) for i in I_plus])+\
        #           np.sum([self.t_min[0]*np.einsum('ij,j',c_j.T,jac.T[i].T) for i in I_minus])
        #     d_.append(d_j)
        # print('d_', d_)
        # print('d', d)
        return -self.mu_t * np.min(min_in_list)

    def cal_cost(self, cables_one_side, cables_other_side, feasible_sets, point_end_effector, pose_0, rotation_cent):
        self.set_jacobian_param(point_end_effector, pose_0, rotation_cent)
        v1 = self.cost_cable_interference(cables_one_side, cables_other_side)
        v2 = self.cost_feasible_points(
            cables_one_side, cables_other_side, feasible_sets)
        v3 = self.r_t_AW(cables_one_side, cables_other_side)
        # print('cost_cable_interference: {}\cost_feasible_points: {}\r_t_AW: {}'.format(v1, v2, v3))
        return v1+v2+v3


def generate_x(number):

    # all random
    # x_uav = np.random.random((number, 8)) * 10- 5  # 4 uav, includes x,y
    # generate random points around the trajectory they walking along
    # Case 1: random step
    x_uav_1_x = -2.55 + np.random.random((number))*5.1-2.55  # -4.25, -0.25
    x_uav_2_x = -2.55 + np.random.random((number))*5.1-2.55  # -1.5, 2.5
    x_uav_3_x = 2.55 + np.random.random((number))*5.1-2.55  # 1, 5
    x_uav_4_x = 2.55 + np.random.random((number))*5.1-2.55  # -2, 2
    x_uav_1_y = 2.55 + np.random.random((number))*5.1-2.55  # -0.3, 3.7
    x_uav_2_y = -2.55 + np.random.random((number))*5.1-2.55  # -4.7, -0.7
    x_uav_3_y = -2.55 + np.random.random((number))*5.1-2.55  # -0.3, 3.7
    x_uav_4_y = 2.55 + np.random.random((number))*5.1-2.55  # -1.8, 2.2

    # Case 2: step = 0.001
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
    x_uav_temp = [x_uav_1_x, x_uav_1_y, x_uav_2_x,
                  x_uav_2_y, x_uav_3_x, x_uav_3_y, x_uav_4_x, x_uav_4_y]
    x_uav = np.array(x_uav_temp).T
    # x_uav_ = []
    # for i in range(len(x_uav_temp)):
    #     uav_temp = x_uav_temp[i]
    #     # print(len(uav_temp))
    #     # np.random.shuffle(uav_temp)
    #     x_uav_.append(copy.deepcopy(uav_temp))
    # # print(x_uav_)
    # x_uav = np.array(x_uav_).T

    #all random
    # param_var = np.random.random((number, 6)) * 2-1 # platform pose x,y,z,r,p,y
    # generate random points around the trajectory they walking along
    # Case 1: random step


    print(np.shape(x_uav))


    return x_uav


if __name__ == "__main__":
    value = Value_function()
    uav_posi = generate_x(100)
    for i in range(10):
        temp = uav_posi[i]
        temp = np.insert(uav_posi[i], [2, 4, 6, 8], [8.0, 8.0, 8.0, 8.0])
        temp = np.reshape(temp,(4,3))
        cost = value.cost_feasible_points(temp)
        # print(temp)
        # print(cost)


    # # CABLE_ONE_SIDE = np.array([[0, 0.29, 4.25],
    # #                            [-0.25, -0.145, 4.25],
    # #                            [0.25, -0.145, 4.25],
    # #                            [0, 0, 4.25],
    # #                            [0, 0.29, 3.75],
    # #                            [-0.25, -0.145, 3.75],
    # #                            [0.25, -0.145, 3.75]])
    # # CABLE_OTHER_SIDE = np.array([[-2.5, 1.45, 8.0],
    # #                              [0, -2.9, 8.0],
    # #                              [2.5, 1.45, 8.0],
    # #                              [0, 0, 8],
    # #                              [2.5, 1.45, 0],
    # #                              [-2.5, 1.45, 0],
    # #                              [0, -2.9, 0]])

    # CABLE_ONE_SIDE = np.array(
    #     [[1.88416284, 1.09104124, 3.67928843] for _ in range(7)])
    # CABLE_OTHER_SIDE = np.array([[1.45588189, - 0.48157763,  7.67122168],
    #                                 [0.66489952, - 1.12164579, 7.64883962],
    #                                 [3.72773982, 3.14699436, 7.68854749],
    #                                 [2.83510081, - 0.75169482, 7.84803509],
    #                                 [2.5,    1.45,  0.],
    #                                 [-2.5,    1.45,    0.],
    #                                 [0., - 2.9,    0.]])
    # print(np.shape(CABLE_ONE_SIDE))
    # pose_0 = [0, 0, 0, 1]
    # rotation_cent = np.array([0, 0, 4])
    # # point_end_effector = np.array(
    # #     [np.array([0.0, 0.29, 0.25]), np.array([-0.25, -0.145, 0.25]), np.array([0.25, -0.145, 0.25]),
    # #      np.array([0.0, 0.0, 0.25]),
    # #      np.array([0.0, 0.29, -0.25]), np.array([-0.25, -0.145, -0.25]), np.array([0.25, -0.145, -0.25])])
    # point_end_effector = np.array([[0.,0.,0.] for _ in range(7)])
    # value.set_jacobian_param(point_end_effector, pose_0, rotation_cent)
    # v = value.cost_cable_interference(CABLE_ONE_SIDE, CABLE_OTHER_SIDE)
    # v1 = value.cost_cable_length(CABLE_ONE_SIDE, CABLE_OTHER_SIDE)
    # v2 = value.cost_feasible_points(
    #     CABLE_ONE_SIDE, CABLE_OTHER_SIDE, np.array([5, 5, 5, 5, 5, 5, 5]))
    # r_t_AW = value.r_t_AW(CABLE_ONE_SIDE, CABLE_OTHER_SIDE)
    # r_r_AW = value.r_r_AW(CABLE_ONE_SIDE, CABLE_OTHER_SIDE, rotation_cent)
    # print('cost_cable_interference: {}\ncost_cable_length: {}\ncost_feasible_points: {}'.format(
    #     v, v1, v2))
    # print('r_r_AW: {}\nr_t_AW: {}'.format(r_r_AW, r_t_AW))

    # from Jacobi import jacobi
    # w = [0, 0, 1, 0, 0, 0]
    # jac, _, _ = jacobi(CABLE_OTHER_SIDE, point_end_effector,
    #                    pose_0, rotation_cent, w)
    # v5 = value.cost_r(jac)
    # print(v5)

    # # value_input = sio.loadmat(
    # #     '/home/firefly/chh_ws/src/plan_cdpr/scripts/data/network_value_input_2021-11-10-20:39:47.mat')

    # # # 'CABLE_ONE_SIDE', 'CABLE_OTHER_SIDE', 'POSE_0', 'ROTATION_CENTER', 'point_end_effector'
    # # print(value_input.keys())
    # # # print(value_input['CABLE_OTHER_SIDE'][0])
    # # plot_r_t = []
    # # plot_r_r = []
    # # for i in range(2,len(value_input['CABLE_OTHER_SIDE'])):
    # #     CABLE_ONE_SIDE = value_input['CABLE_ONE_SIDE'][i]
    # #     CABLE_OTHER_SIDE = value_input['CABLE_OTHER_SIDE'][i]
    # #     rotation_cent = value_input['ROTATION_CENTER'][i]
    # #     pose_0 = value_input['POSE_0'][i]
    # #     # print(value_input['CABLE_OTHER_SIDE'][2])
    # #     # print('v', value_input['ROTATION_CENTER'][2])
    # #     # print(np.shape(CABLE_ONE_SIDE))
    # #     # print(np.shape(CABLE_OTHER_SIDE))

    # #     value.set_jacobian_param(point_end_effector, pose_0, rotation_cent)
    # #     v = value.cost_cable_interference(CABLE_ONE_SIDE, CABLE_OTHER_SIDE)
    # #     v1 = value.cost_cable_length(CABLE_ONE_SIDE, CABLE_OTHER_SIDE)
    # #     v2 = value.cost_feasible_points(CABLE_ONE_SIDE, CABLE_OTHER_SIDE, np.array([5, 5, 5, 5, 5, 5, 5]))
    # #     r_t_AW = value.r_t_AW(CABLE_ONE_SIDE, CABLE_OTHER_SIDE)
    # #     r_r_AW = value.r_r_AW(CABLE_ONE_SIDE, CABLE_OTHER_SIDE, rotation_cent)
    # #     print('cost_cable_interference: {}\ncost_cable_length: {}\ncost_feasible_points: {}'.format(
    # #         v, v1, v2))
    # #     print('r_r_AW: {}\nr_t_AW: {}'.format(r_r_AW, r_t_AW))
    # #     plot_r_r.append(-r_r_AW/1.0)
    # #     plot_r_t.append(-r_t_AW/20.0)
    # # plt.subplot(211)
    # # plt.plot(plot_r_r, 'r')
    # # plt.legend(labels=['r_r_AW'], loc='best')

    # # plt.subplot(212)
    # # plt.plot(plot_r_t, 'g')
    # # plt.legend(labels=['r_t_AW'], loc='best')
    # # plt.savefig('../scripts/figure/r_.png')

    # # plt.show()

