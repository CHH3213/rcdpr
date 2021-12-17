# -*-coding:utf-8-*-
"""
compute jacobi---from matlab convert
returns f,w
"""
import numpy as np
import scipy
from scipy.optimize import linprog
from rotate_calculation import Rotate
from module_test.null_test import *

def rotz(ang):
    rad = ang / 180 * np.pi
    return np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])


def jacobi(points_base=None, point_end_effector=None, pose_0=None, rotation_center=None, w=None):
    """
    计算雅可比与绳子拉力
    :param w: 给定的力矩
    :param points_base: 相当于无人机和小车上的 固定点（基坐标系）
    :param point_end_effector: 移动平台(末端执行器）固定点（自身坐标系）
    :param pose_0: 末端执行器姿态四元数
    :param rotation_center:末端执行器坐标系相对于基坐标系的位置（即论文中点P的位置）
    :return:
    """
    # wrench and tension
    r = Rotate()
    f = np.zeros((7, 1))
    point_end_effector_baseframe = np.empty((7, 3))
    points_base = np.array(points_base)
    point_end_effector = np.array(point_end_effector)
    pose_0 = np.array(pose_0)
    rotation_center = np.array(rotation_center)
    w = np.array(w)

    for i in range(0, 7, 1):
        point_end_effector_baseframe[i] = rotation_center

    b = point_end_effector_baseframe - rotation_center  # 相对机体坐标系
    b = b.T
    # print('b_', b)
    # 论文中的q，即从A到B的向量，并对其单位化
    q = points_base - point_end_effector_baseframe
    # print('q',q)
    q = q.T
    # print('q_',q)
    '''======================'''
    # get jacobian
    u = np.zeros((3, 7))
    J = np.zeros((3, 7))
    for i in range(7):
        u[:, i] = q[:, i] / np.linalg.norm(q[:, i])
        J[:, i] = u[:, i] # *号为解开
    # print(np.shape(u))
    # print(np.shape(J))  # 6*7
    J = -J.T
    # print('J', J.T)
    # print(np.shape(J))
    # 奇异值分解并进行基底变换 FIXME: 和matlab的输出不一样
    U, S, V = np.linalg.svd(-J.T)
    # print(np.einsum('ij,ji->i',U,U.T))
    # print(np.einsum('ij,ji->i',V,V.T))
    # Sbar = S[:,:6]
    Sbar = S
    wu = np.einsum('ij,j->i', U.T, w)
    fvabr = wu / Sbar  # 特解
    Vinv = V.T
    # print('fvabr: {}\tVinv:{}'.format(fvabr, Vinv))
    # print(np.shape(Vinv))
    f1 = np.einsum('ij,j->i', Vinv[:,0:3], fvabr)  # 特解回到原来基底


    # TODO: 计算通解
    null_ = null(-J.T)

    # TODO: linear program
    c = [0,0,0,0,1]
    Aub_1 = np.concatenate((null_,-np.ones((7,1))),axis=1)
    Aub_2 = np.concatenate((-null_, -np.zeros((7, 1))), axis=1)
    Aub = np.concatenate((Aub_1,Aub_2))
    # print('Aub:',Aub)
    bub = np.concatenate((-f1,f1),axis=0)
    # print('bub: ',bub)
    bounds = [(None, None) for i in range(5)]
    res = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds)
    x = np.array(res.x[0:4])
    f = np.einsum('ij,j->i',null_,x) + f1
    
    
    fw = np.einsum('ij,j->i', -J.T, f)  # 检测是否和输入的力螺旋一样
    # print('f:{}'.format(f))
    # print('fw:{}'.format(fw))
    # if fw[0]-0<1e-10:
    #     print(fw[0])
    #     print('hh')
    # assert np.all(fw)== np.all(np.array([0, 0, 2.5, 0, 0, 0]))
    return -J.T, f, fw


def torque(J, f):
    '''
    检测计算是否正确
    :param J: 雅可比
    :param f: 拉力
    :return: 力螺旋
    '''
    tor = J @ f
    print('torque', tor)
    return tor


if __name__ == '__main__':
    # jacobi()  # 原来设定值
    points_base = np.array([[0, 4, 7.5], [3.464, -2, 7.5], [-3.464, -2, 7.5],
                            [0, 0, 7.5], [0, 2, 0], [1.732, -1, 0], [-1.732, -1, 0]])
    # point_end_effector = np.array(
    #     [[0.0, 0.289, 0.05], [0.25, -0.144, 0.05], [-0.25, -0.144, 0.05],
    #      [0.0, 0.0, 0.05],
    #      [0.0, 0.289, -0.05], [0.25, -0.144, -0.05], [-0.25, -0.144, -0.05]])
    point_end_effector = np.zeros((7,3))
    # print(np.shape(point_end_effector))
    pose_0 = [0, 0, 0, 1]
    rotation_center = [0, 0, 0]
    w = [0, 0, 1]
    j, f, fw = jacobi(points_base, point_end_effector,
                     pose_0, rotation_center, w)
    print(fw)
    # torque(j, f)
    # print('j',j)
    # print('f',f)
