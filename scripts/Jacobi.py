# -*-coding:utf-8-*-
"""
compute jacobi---from matlab convert
returns f,w
"""
import numpy as np
import scipy
from rotate_calculation import Rotate
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
    '''测试'''
    # w = np.array([1, 1, 1, 10, 1, 1])  # 力螺旋
    # # anchor point
    # a = np.array([10, 1, 3])
    # b = np.array([0, 5, 1])
    # a1 = np.einsum('ij,j->i', rotz(0), a)
    # a2 = np.einsum('ij,j->i', rotz(120), a1)
    # a3 = np.einsum('ij,j->i', rotz(-120), a1)
    # a4 = np.array([a1[0], a1[1], -a1[2]])
    # a5 = np.einsum('ij,j->i', rotz(120), a4)
    # a6 = np.einsum('ij,j->i', rotz(-120), a4)
    # a7 = np.array([0, 0, a[2]])
    # a = np.array([a1, a2, a3, a4, a5, a6, a7]).T
    # b1 = np.einsum('ij,j->i', rotz(60), b)
    # b2 = np.einsum('ij,j->i', rotz(120), b1)
    # b3 = np.einsum('ij,j->i', rotz(-120), b1)
    # b4 = np.einsum('ij,j->i', rotz(-60), np.array([b[0], b[1], -b[2]]))
    # b5 = np.einsum('ij,j->i', rotz(120), b4)
    # b6 = np.einsum('ij,j->i', rotz(-120), b4)
    # b7 = np.array([0, 0, b[2]])
    # b = np.array([b1, b2, b3, b4, b5, b6, b7]).T
    # q = a-b
    # print(np.shape(q))  #3*7
    '''覆写'''
    # quaternion rotation of a vector
    for i in range(0, 7, 1):
        # TODO:是否有问题？导致算出的f负的
        vector = np.hstack([point_end_effector[i], 0])  # 将B点转换为四元数
        # 通过四元数的旋转变换求得旋转后B’的四元数位置（相对于末端执行器坐标系）
        # print('vector', vector)
        rotated_vector = r.rotated_vector(pose_0,vector)
        # print('rotated_vector', rotated_vector)  # 经检查没问题
        # 将B’的四元数位置变成3维位置并进行坐标补偿将它变成相对于基坐标系的位置
        point_end_effector_baseframe[i] = np.delete(rotated_vector, 3) + rotation_center
        # print('point_end_effector_baseframe',point_end_effector_baseframe[i])
    b = point_end_effector_baseframe - rotation_center  # 相对机体坐标系
    # print('b', b)
    # print(np.shape(b))
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
    J = np.zeros((6, 7))
    for i in range(7):
        u[:, i] = q[:, i] / np.linalg.norm(q[:, i])
        # print(np.shape(np.cross(b[:, i], u[:, i])))
        # print(np.shape(u[:, i]))
        # # print(*u[:, i])
        # print(np.shape(J[:,i]))
        J[:, i] = np.array([*u[:, i], *np.cross(b[:, i], u[:, i])])  # *号为解开
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
    f1 = np.einsum('ij,j->i', Vinv[:, 0:6], fvabr)  # 特解回到原来基底
    # print(f1)

    kmin = -np.inf
    kmax = np.inf
    v7 = np.array(Vinv[:, 6])  # %  分解出列零空间基底，其余为列空间基底
    # print('v7:{}'.format(v7))
    for i in range(7):
        if abs(v7[i]) < 1 / np.inf:
            if f1[i] >= -1 / np.inf:
                pass  # 认为 >= 0
            else:
                print('infeasible')
        else:
            if v7[i] > 1e-10:  # v7在该维度>0，k不能太小，使得f相应维度<0（下同）
                imin = (0 - f1[i]) / v7[i]
                if imin > kmin:
                    kmin = imin  # %   更新k下限

            else:
                imax = (0 - f1[i]) / v7[i]
                if imax < kmax:
                    kmax = imax  # %   更新k上限
    # % 最大值最小
    # % 有解时kmin, kmax至少有一个成功更新
    if not (kmin == -np.inf):
        if not (kmax == np.inf):
            # if kmin >= kmax:
            #     print('infeasible')
            fmax_kmin = max(f1 + kmin * v7)
            fmax_kmax = max(f1 + kmax * v7)
            if fmax_kmin > fmax_kmax:
                k = kmax
            else:
                k = kmin
        else:
            k = kmin
    else:
        k = kmax
    # print(kmin, kmax)
    f = f1 + k * v7
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
    point_end_effector = np.array(
        [[0.0, 0.289, 0.05], [0.25, -0.144, 0.05], [-0.25, -0.144, 0.05],
         [0.0, 0.0, 0.05],
         [0.0, 0.289, -0.05], [0.25, -0.144, -0.05], [-0.25, -0.144, -0.05]])
    # print(np.shape(point_end_effector))
    pose_0 = [0, 0, 0, 1]
    rotation_center = [0, 0, 0]
    w = [0, 0, 1, 0, 0, 0]
    j, f, _ = jacobi(points_base, point_end_effector, pose_0, rotation_center, w)
    torque(j, f)
    # print('j',j)
    # print('f',f)
