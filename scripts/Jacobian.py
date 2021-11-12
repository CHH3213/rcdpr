# -*-coding:utf-8-*-
'''
他雅可比矩阵计算
'''
import numpy as np
# import tf
from rotate_calculation import Rotate
class JacobianAndForce(object):
    """
    Superclass for environments.
    """
    def __init__(self):
        self.b_X_u = np.empty((7,3)) # 7跟绳子，每个点是 3惟的
        self.point_end_effector_baseframe = np.empty((7,3)) # 7个固定点，每个点是 3惟的
        # print self.b_X_u
        self.r = Rotate()

    def get_jacobian(self, points_base, point_end_effector, pose_0, rotation_center):
        '''
        参数分别为在基坐标系上的固定点A位置（相对于基坐标系），在末端执行器上的固定点B位置（相对于末端执行器坐标系），
        表示末端执行器位姿的四元数，末端执行器坐标系相对于基坐标系的位置（即论文中点P的位置）
        :param points_base: 相当于无人机和小车上的 固定点（基坐标系）
        :param point_end_effector: 移动平台(末端执行器）固定点（自身坐标系）
        :param pose_0: 末端执行器姿态四元数
        :param rotation_center：末端执行器坐标系相对于基坐标系的位置（即论文中点P的位置）
        :return: jacobian
        '''
        # quaternion rotation of a vector
        for i in range(0, 7, 1):
            vector = np.hstack([point_end_effector[i], 0])  # 将B点转换为四元数
            # 通过四元数的旋转变换求得旋转后B’的四元数位置（相对于末端执行器坐标系）
            rotated_vector =self.r.rotated_vector(pose_0,vector) 
            # 将B’的四元数位置变成3维位置并进行坐标补偿将它变成相对于基坐标系的位置
            self.point_end_effector_baseframe[i] = np.delete(rotated_vector, 3) + rotation_center
            # print('rotated_vector', rotated_vector)  # 经检查没问题
            # print('point_end_effector_baseframe', self.point_end_effector_baseframe[i])

        b = self.point_end_effector_baseframe - rotation_center
        # print('b',b)
        # 论文中的u，即从A到B的向量，并对其单位化
        u = points_base - self.point_end_effector_baseframe
        # print('u',u)
        for i in range(0, 7, 1):
            u[i] = u[i]/np.linalg.norm(u[i])
        # 根据论文所给的公式得到雅可比矩阵（可能少了论文中的负号，只需在后续计算时注意即可）
        for i in range(0, 7, 1):
            self.b_X_u[i] = np.cross(b[i], u[i])
            # print np.cross(b[i],u[i])
            # print self.b_X_u

        J = np.row_stack((u.T, self.b_X_u.T))
        return J


    def cable_force(self, points_base, point_end_effector, pose_0, rotation_center, target_torque):
        '''
        该函数用于得到各绳子需要施加的拉力，主要是用了动力学方程根据T = J^(-1)*w ，然后解线性方程组。
        通解+特解
        最后对求解出来的负数力进行了一个补偿。（其实6根绳子就能控制6个自由度，只是为了达到完全约束加了一个补偿力）
        :param points_base: 相当于无人机上的 固定点（基坐标系）
        :param point_end_effector: 移动平台(末端执行器）固定点（自身坐标系）
        :param pose_0: 末端执行器姿态四元数
        :param target_torque: PID得到
        :return:
        '''
        J = self.get_jacobian(points_base, point_end_effector, pose_0, rotation_center)
        j_rotation = J
        equivalence_vector = np.empty(7)
        cable_force_vector = np.empty(7)
        # print cable_force_vector
        # qiu通解  这边：J=[J1(6*6),J2(6*1)]：6*7， t=[t1(6*1) t2(1*1)]：7*1， w：6*1  J1*t1+J2*t2 = 0,默认t2=1
        cable_others = j_rotation[:, 0:6]  # 6*6矩阵
        cable_one = -j_rotation[:, 6]   # 6*1矩阵
        equivalence = np.linalg.solve(cable_others, cable_one)  # equivalence = (cable_others)^(-1)*cable_one
        for i in range(0, 6, 1):
            equivalence_vector[i] = equivalence[i]
        equivalence_vector[6] = 1
        print('v',equivalence_vector)
        # 求特解
        cable_force = np.linalg.solve(cable_others, target_torque)
        for i in range(0, 6, 1):
            cable_force_vector[i] = cable_force[i]
        cable_force_vector[6] = 0

        for i in range(0, 7, 1):
            if (cable_force_vector[i]<1e-10):
                cable_force_vector = cable_force_vector + equivalence_vector * (-cable_force_vector[i]/equivalence_vector[i])

        # cable_force_vector = np.linalg.lstsq(J,target_torque)  # 测试
        # cable_force_vector = np.dot(np.linalg.pinv(J), target_torque)  # 测试2

        return cable_force_vector

    def torque(self, points_base, point_end_effector, pose_0, action):
        '''
        根据w = J×T计算扭矩
        :param points_base: 相当于无人机上的 固定点（基坐标系）
        :param point_end_effector: 移动平台(末端执行器）固定点（自身坐标系）
        :param pose_0: 末端执行器姿态四元数
        :param action: 绳子张力
        :return:
        '''
        J = self.get_jacobian(points_base, point_end_effector, pose_0)
        j_rotation = J
        torque = np.dot(j_rotation,np.reshape(action,(7,1)))  # w = J*T

        return torque

if __name__=='__main__':
    Jac = JacobianAndForce()
    points_base = np.array([[-2.5, 1.45, 8.0],
                             [0, -2.9, 8.0],
                             [2.5, 1.45, 8.0],
                             [0, 0, 8],
                             [2.5, 1.45, 0],
                             [-2.5, 1.45, 0],
                             [0, -2.9, 0]])
    point_end_effector = np.array(
        [np.array([0.0, 0.29, 0.25]), np.array([-0.25, -0.145, 0.25]), np.array([0.25, -0.145, 0.25]),
         np.array([0.0, 0.0, 0.25]),
         np.array([0.0, 0.29, -0.25]), np.array([-0.25, -0.145, -0.25]), np.array([0.25, -0.145, -0.25])])
    # print(np.shape(point_end_effector))
    pose_0 = [0, 0, 0, 1]
    rotation_center = [0, 0, 4]
    w = [0, 0, 1, 0, 0, 0]
    J = Jac.get_jacobian(points_base,point_end_effector,pose_0,rotation_center)
    print('J', J)
    force = Jac.cable_force(points_base,point_end_effector,pose_0,rotation_center,w)
    print('f',force)
