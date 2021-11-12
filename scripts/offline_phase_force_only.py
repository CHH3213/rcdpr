# -*-coding:utf-8-*-
'''
4架无人机，3辆小车
2021/11/5 使用他优化出来的力收集轨迹数据
通过代价函数获取数据，离线训练神经网络
'''
from __future__ import print_function, absolute_import, division

import rospy
from std_msgs.msg import Float32MultiArray
import os
from PIDClass import PID
import gym
import tf
import scipy.io as sio
from gazebo_msgs.srv import *
import numpy as np
import matplotlib.pyplot as plt
from control_drone import Drone
from control_car import Omni_car
from gazebo_reset import Gazebo_reset
from gradient_calculation import Gradient_calculation
from rotate_calculation import Rotate
from value_function import Value_function
from Jacobian import JacobianAndForce as Jacobia
from Jacobi import jacobi
from module_test import replay_buffer
from optimizer import *
from pidController import *
import time



class CDPR(gym.Env):
    def __init__(self):
        # # 物理属性
        self.payload_gravity = 1.0
        self.drone_gravity = 18.88
        self.dt = 0.1  ## 仿真频率
        self.save_data_dir = "../scripts/data"
        if not os.path.exists(self.save_data_dir):
            os.makedirs(self.save_data_dir)

        self.force_publisher0 = rospy.Publisher('/rcdpr_force', Float32MultiArray, queue_size=10)

        ### Initiate ROS node
        print('-- Connecting to mavros')
        rospy.init_node('node', anonymous=True)
        print('connected')
        self.Drone1 = Drone('drone1')
        self.Drone2 = Drone('drone2')
        self.Drone3 = Drone('drone3')
        self.Drone4 = Drone('drone4')
        self.Car0 = Omni_car('omni_car_0')
        self.Car1 = Omni_car('omni_car_1')
        self.Car2 = Omni_car('omni_car_2')
        self.gazebo_reset = Gazebo_reset()
        # self.Gradient = Gradient_calculation()
        self.Rot = Rotate()
        self.J = Jacobia()
        # TODO:参数的具体定义
        self.m = self.payload_gravity / 9.8
        self.d_min = 0.1
        self.mu_t, self.mu_r, self.mu_d, self.mu_a, self.mu_l = 1.0, 1.0, 0.017, 1.0, 0.2
        self.t_min = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.t_max = np.array([5, 5, 5, 5, 5, 5, 5])
        self.target_length_list = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
        self.value_function = Value_function(self.m, d_min=self.d_min, mu_t=self.mu_t,
                                             mu_r=self.mu_r, mu_d=self.mu_d, mu_a=self.mu_a,
                                             t_min=self.t_min, t_max=self.t_max, mu_l=self.mu_l,
                                             target_length_list=self.target_length_list)
        # 末端执行器固定点在其自身坐标系中的位置uav1~4,ugv1~3
        self.point_end_effector = np.array([[0, 0, 0] for _ in range(7)])
        self.cable_number = 7  # 绳子数量
        self.cable_length = np.array([5 for _ in range(self.cable_number)])
        self.cable_one_side = np.empty((7, 3))  # 7个固定点，每个点是 3惟的

        # value function 需要保存的输入输出数据
        self.input = []
        self.input_platform = []
        self.output = []

        # reset
        self.gazebo_reset.reset_Car_State([self.Car0, self.Car1, self.Car2])
        self.gazebo_reset.resetPayloadState()

        # 经验池
        self.replay_buff = replay_buffer.ReplayBuffer(1)

        # 画图数据
        self.wrench = []
        self.t = []
        self.time_step = []
        self.platform_pos = []

    def run(self):
        '''
        运行代价函数收集数据
        :return:
        '''

        self.gazebo_reset.resetPayloadState()
        time.sleep(0.1)
        self.gazebo_reset.reset_Car_State([self.Car0, self.Car1, self.Car2])
        time.sleep(0.1)

        # logger and UAV
        multiarray0 = Float32MultiArray()

        hz = 5
        r = rospy.Rate(hz)
        count_draw = 1
        # 初始平台位置
        self.dist_x = 0
        self.dist_y = 0
        self.dist_z = 3.5
        dist_x = 0  # 小车误差修正
        dist_y = 0
        # 圆形轨迹圆心和半径长
        radius = 1
        c0 = [1, 0]
        theta = 0
        total_steps = 2000
        '''
        无人机解锁并起飞
        '''
        self.Drone1.set_mode('GUIDED')
        self.Drone2.set_mode('GUIDED')
        self.Drone3.set_mode('GUIDED')
        self.Drone4.set_mode('GUIDED')
        self.Drone1.set_arm(True)
        self.Drone2.set_arm(True)
        self.Drone3.set_arm(True)
        self.Drone4.set_arm(True)
        rospy.sleep(1)
        self.Drone1.takeoff(8)
        self.Drone2.takeoff(8)
        self.Drone3.takeoff(8)
        self.Drone4.takeoff(8)
        rospy.sleep(5)
        self.Drone1.goto_xyz(0.87, 1.5, 7.5)
        self.Drone2.goto_xyz(-1.74, 0, 7.5)
        self.Drone3.goto_xyz(0.87, -1.5, 7.5)
        self.Drone4.goto_xyz(2, 0, 7.5)
        rospy.sleep(6)
        self.count = 0
        # TODO:payload初始化
        self.gazebo_reset.set_payload_state()  # 将payload初始化到(0,0,3)
        while not rospy.is_shutdown():
            # 画圆形轨迹 (x-x0)^2+(y-y0)^2=r^2  <==> 令圆心为（2，0），半径为2，==》x = 2cos(pi-theta)+2,  y=sin(pi-theta),在1000步时刚好画完一个圆
            start = time.time()
            '''=====================轨迹=========================='''
            if count_draw > 500:
                theta += 2 * np.pi / (total_steps - 500)
                # print('theat',theta)
                temp = [dist_x, dist_y]
                self.dist_x = radius * np.cos(np.pi - theta) + c0[0]
                self.dist_y = radius * np.sin(np.pi - theta) + c0[1]
                dist_x = 1.1 * (radius * np.cos(np.pi - theta) + c0[0])  # 小车误差修正
                dist_y = 1.1 * radius * np.sin(np.pi - theta) + c0[1]
                # print('temp',temp)
                # print('dist',dist_x,dist_y)
                temp_pos = [dist_x - temp[0], dist_y - temp[1]]  # 质点一步要走的距离
                # print('distance', temp_pos)
                # 小车运动
                # self.Car0.goto_car_xyz(temp_pos, self.logger0_attitude, self.dt)
                # # time.sleep(0.0005)
                # self.Car1.goto_car_xyz(temp_pos, self.logger1_attitude, self.dt)
                # # time.sleep(0.0005)
                # self.Car2.goto_car_xyz(temp_pos, self.logger2_attitude, self.dt)
            '''==================================='''
            # 无人机的机体坐标刚好和世界坐标反过来了
            self.Drone1.goto_xyz(-self.dist_y + 0.87, self.dist_x + 1.5, 7.5)
            self.Drone2.goto_xyz(-self.dist_y - 1.74, self.dist_x, 7.5)
            self.Drone3.goto_xyz(-self.dist_y + 0.87, self.dist_x - 1.5, 7.5)
            self.Drone4.goto_xyz(-self.dist_y + 2, self.dist_x, 7.5)

            '''######### 质点位置目标  ###########'''
            # ***********************************
            target = [self.dist_x, self.dist_y, self.dist_z]
            # print('payload_target', target)
            pid = apmPID(target)
            # uav:1~4,ugv:1~3
            # TODO:计算得到的绳子拉力大小应该有问题
            cable_tensions = self.compute_cable_tensions(self.payload_gravity, self.drone_gravity, pid)
            ########################################################
            # drone--force
            # TODO:这里不该再有cable_tension的非负限制
            for i in range(len(cable_tensions)):
                if cable_tensions[i] < 0:
                    cable_tensions[i] = 0
            # cable_tensions = np.where(cable_tensions > 0, cable_tensions, 0)

            force_payload2drone1 = min(cable_tensions[0], 5)
            force_payload2drone2 = min(cable_tensions[1], 5)
            force_payload2drone3 = min(cable_tensions[2], 5)
            force_payload2drone4 = min(cable_tensions[3], 5)
            # 小车共用一个话题
            force_logger0 = min(cable_tensions[4], 5)
            force_logger1 = min(cable_tensions[5], 5)
            force_logger2 = min(cable_tensions[6], 5)

            multiarray0.data = [force_logger0 * 1000 / hz, force_logger1 * 1000 / hz, force_logger2 * 1000 / hz,
                                force_payload2drone1 * 1000 / hz, force_payload2drone2 * 1000 / hz,
                                force_payload2drone3 * 1000 / hz, force_payload2drone4 * 1000 / hz]
            # 无人机共用一个话题
            # print('pos', self.drone1_position[2])
            if self.drone1_position[2] > 0 and self.drone2_position[2] > 0 and self.drone3_position[2] > 0 and \
                    self.drone4_position[2] > 0:
                self.force_publisher0.publish(multiarray0)

            '''生成代价函数数据'''
            self.generate_train_data()
            '''-----------------------------------'''
            r.sleep()
            self.dt = time.time() - start
            print('compute_time', time.time() - start)
            count_draw += 1
            self.time_step.append(count_draw)
            if count_draw == total_steps:
                sio.savemat(self.save_data_dir + '/value_data.mat',
                            dict(input_platform=self.input_platform, input_pos=self.input, output_value=self.output))
                # sio.savemat(self.save_data_dir + '/plot_data.mat',
                #             dict(steps=self.time_step, platform_pos=self.platform_pos, wrench=self.wrench,
                #                  tensions=self.t))
                self.plot_curve()
                break
                
            hz = 1.0/(time.time()-start)
        self.Drone1.set_mode('RTL')
        self.Drone2.set_mode('RTL')
        self.Drone3.set_mode('RTL')
        self.Drone4.set_mode('RTL')

        self.gazebo_reset.reset_Car_State([self.Car0, self.Car1, self.Car2])
        self.gazebo_reset.resetPayloadState()
        print('结束！')

    def compute_cable_tensions(self, payload_gravity, drone_gravity, pid):
        '''
        :param cableForce1: 无人机1推力
        :param cableForce2: 无人机2推力
        :param payload_gravity: 负载重力
        :param pid: pid控制器
        :return: 返回每根绳子的拉力大小
        '''
        [self.payload_position, self.payload_attitude, self.payload_pose0, self.payload_linear, _] = self.get_state(
            self.gazebo_reset.model_states, 'payload')
        [self.logger0_position, self.logger0_attitude, _, _, _] = self.get_state(self.gazebo_reset.model_states,
                                                                                 'omni_car_0')
        [self.logger1_position, self.logger1_attitude, _, _, _] = self.get_state(self.gazebo_reset.model_states,
                                                                                 'omni_car_1')
        [self.logger2_position, self.logger2_attitude, _, _, _] = self.get_state(self.gazebo_reset.model_states,
                                                                                 'omni_car_2')
        # 使用mavros获取无人机位置信息
        # [self.drone1_position, ori_drone1] = self.Drone1.get_posByMavros()
        [self.drone1_position, ori_drone1, _, _, _] = self.get_state(self.gazebo_reset.model_states, 'drone1')
        [self.drone2_position, ori_drone2, _, _, _] = self.get_state(self.gazebo_reset.model_states, 'drone2')
        [self.drone3_position, ori_drone3, _, _, _] = self.get_state(self.gazebo_reset.model_states, 'drone3')
        [self.drone4_position, ori_drone4, _, _, _] = self.get_state(self.gazebo_reset.model_states, 'drone4')

        #  pid controller: 目标位置+姿态-->目标力和力矩
        # TODO:PID调整参数
        target_force_wrench = pid.cal_actions(self.payload_position )
        compensate_wrench = np.array([0, 0, self.payload_gravity])  # 补偿负载重力
        target_force_wrench = np.array(target_force_wrench) + compensate_wrench
        # print('target_wrench', target_force_wrench)
        # agent固定点位置：
        self.points_base = np.array(
            [np.array(self.drone1_position), np.array(self.drone2_position), np.array(self.drone3_position),
             np.array(self.drone4_position),
             np.array(self.logger0_position), np.array(self.logger1_position), np.array(self.logger2_position)])
        # 末端执行器上固定点位置（相对于自身）
        point_end_effector = self.point_end_effector
        # 末端执行器姿态--四元数
        pose_0 = np.array(self.payload_pose0)
        # 末端执行器坐标系相对于基坐标系的位置
        rotation_center = np.array(self.payload_position)

        for i in range(0, 7, 1):
            vector = np.hstack([point_end_effector[i], 0])  # 将B点转换为四元数
            # 通过四元数的旋转变换求得旋转后B’的四元数位置（相对于末端执行器坐标系）
            rotated_vector = self.Rot.rotated_vector(pose_0, vector)
            # 将B’的四元数位置变成3维位置并进行坐标补偿将它变成相对于基坐标系的位置
            self.cable_one_side[i] = np.delete(rotated_vector, 3) + rotation_center

        # TODO: 雅喀比 --> 两个雅可比的绳子张力不一样
        self.Jac = self.J.get_jacobian(self.points_base, point_end_effector, pose_0, rotation_center)
        # cable_tensions = self.J.cable_force(self.points_base, point_end_effector, pose_0, rotation_center,
        #                                     target_force_wrench)
        # print('J1', self.Jac[0])
        self.Jac, cable_tensions, force_wrench = jacobi(self.points_base, point_end_effector, pose_0, rotation_center,
                                                        target_force_wrench)
        '''save data'''
        self.wrench.append(target_force_wrench)
        self.t.append(cable_tensions)
        self.platform_pos.append(self.payload_position)
        # print('J2', self.Jac[0])
        # print('===========================')
        # print(np.where(cable_tensions < -(1e-15)))
        is_empty = np.where(cable_tensions < -1e-15)
        # print(type(is_empty))  #tuple
        is_empty = np.array(is_empty)
        # print(is_empty)
        if is_empty.size > 0:
            self.count += 1
            print('number of cable_tensions<0:', self.count)
            print('cable_tensions', cable_tensions)
        #     cable_tensions = self.replay_buff.sample(1)
        # else:
        #     self.replay_buff.push(cable_tensions)
        # # 测试是否还有小于0的数
        # is_empty = np.where(cable_tensions < -1e-15)
        # is_empty = np.array(is_empty)
        # if is_empty.size > 0:
        #     print('cable_tensions', cable_tensions)
        '''optimize method--只能控制力'''
        uav_dir0 = self.force_direction(self.points_base[0], self.cable_one_side[0])
        uav_dir1 = self.force_direction(self.points_base[1], self.cable_one_side[1])
        uav_dir2 = self.force_direction(self.points_base[2], self.cable_one_side[2])
        uav_dir3 = self.force_direction(self.points_base[3], self.cable_one_side[3])
        ugv_dir1 = self.force_direction(self.points_base[4], self.cable_one_side[4])
        ugv_dir2 = self.force_direction(self.points_base[5], self.cable_one_side[5])
        ugv_dir3 = self.force_direction(self.points_base[6], self.cable_one_side[6])
        # # print(uav_dir0)
        args = (target_force_wrench[0:3],uav_dir0,uav_dir1,uav_dir2,uav_dir3,ugv_dir1,ugv_dir2,ugv_dir3)
        cable_tensions, _ = minimizeForce(args)

        return cable_tensions

    def force_direction(self, end_point, start_point):
        '''
        得到力的方向
        :param end_point: 力的终止点 array数组 类型（3，）
        :param start_point: ；力的开始作用点 array数组 类型（3，）
        :return:力的方向即绳子的方向，array数组，类型（3，）
        '''
        end_point = np.array(end_point)
        start_point = np.array(start_point)
        pos_dir = end_point - start_point
        dir = pos_dir / np.linalg.norm(pos_dir)
        return dir

    def get_state(self, model_state, model_name):
        model = model_state.name.index(model_name)  #

        model_pose = model_state.pose[model]
        model_twist = model_state.twist[model]
        model_position = [model_pose.position.x, model_pose.position.y, model_pose.position.z]
        # 四元数
        model_pose0 = [model_pose.orientation.x, model_pose.orientation.y, model_pose.orientation.z,
                       model_pose.orientation.w]
        roll, pitch, yaw = self.Rot.quaternion_to_euler(model_pose.orientation.x, model_pose.orientation.y,
                                                        model_pose.orientation.z, model_pose.orientation.w)
        # 欧拉角
        model_attitude = [roll, pitch, yaw]
        model_linear = [model_twist.linear.x, model_twist.linear.y, model_twist.linear.z]
        model_angular = [model_twist.angular.x, model_twist.angular.y, model_twist.angular.z]
        # print([model_position,model_orientation,model_linear,model_angular])
        # 位置，姿态，线速度，角速度
        return [model_position, model_attitude, model_pose0, model_linear, model_angular]

    def generate_train_data(self):
        '''
        使用8+6维度作为输入
        生成代价函数数据
        :return:
        '''
        # 代价函数处理
        cables_other_side = self.points_base  # 绳子ling一端--agent固定点
        # print(np.shape(self.cable_one_side))  #7*3
        # cable_lines = np.array([np.array([self.cable_one_side[i], cables_other_side[i]]) for i in range(len(self.cable_one_side))])
        # print('cable', cable_lines)
        # print(len(cable_lines))
        cost_cable_interference = self.value_function.cost_cable_interference(self.cable_one_side,
                                                                              cables_other_side)
        feasible_sets = self.cable_length
        cost_feasible_points = self.value_function.cost_feasible_points(self.cable_one_side, cables_other_side,feasible_sets, None)
        '''=========获取可行力空间代价数据，用于训练============'''
        U = self.Jac[0:3]
        bXu = self.Jac[3:]
        # print(np.shape(U))
        # print(np.shape(bXu))
        # d1, u_null1 = self.value_function.AW_part(U, 'U')
        # d2, u_null2 = self.value_function.AW_part(bXu, 'bXu')
        # r_t_AW = self.value_function.r_AW(d1, u_null1)
        # r_r_AW = self.value_function.r_AW(d2, u_null2)
        # print('r_t_AW', r_t_AW)
        # print('r_r_AW', r_r_AW)
        # cost_wrench = self.value_function.cost_wrench_function([r_t_AW, r_r_AW])
        r_t_AW = self.value_function.r_t_AW(self.cable_one_side,cables_other_side)
        r_r_AW = self.value_function.r_r_AW(self.cable_one_side,cables_other_side, self.payload_position)
        cost_wrench =r_t_AW +  r_r_AW

        cost_cable_length = self.value_function.cost_cable_length(self.cable_one_side, cables_other_side)
        cost = cost_feasible_points + cost_cable_interference  + cost_wrench
        input_platform = np.array([self.payload_position, self.payload_attitude])
        # print('cost_cable_length', cost_cable_length)
        # print('cost_feasible_points', cost_feasible_points)
        # print('cost_cable_interference', cost_cable_interference)
        # input_ = self.points_base.flatten()  # 这边处理了那么神经网络拟合时就不需要处理
        input_ = np.array([self.drone1_position,self.drone2_position,self.drone3_position,self.drone4_position])
        new_input = np.delete(input_, -1, axis=1)
        self.input.append(new_input)
        self.input_platform.append(input_platform)
        self.output.append(cost)
    def plot_curve(self):
        plot_wrench = np.array(self.wrench).T
        plot_t = np.array(self.t).T
        plot_pos = np.array(self.platform_pos).T
        plt.figure()
        plt.subplot(321)
        plt.plot(self.time_step, plot_wrench[0])
        plt.legend(labels=['w1'], loc='best')
        plt.subplot(322)
        plt.plot(self.time_step, plot_wrench[1])
        plt.legend(labels=['w2'], loc='best')
        plt.subplot(323)
        plt.plot(self.time_step, plot_wrench[2])
        plt.ylim((-1, 3))
        plt.legend(labels=['w3'], loc='best')
        plt.subplot(324)
        plt.plot(self.time_step, plot_wrench[3])
        plt.legend(labels=['w4'], loc='best')
        plt.subplot(325)
        plt.plot(self.time_step, plot_wrench[4])
        plt.legend(labels=['w5'], loc='best')
        plt.subplot(326)
        plt.plot(self.time_step, plot_wrench[5])
        plt.legend(labels=['w6'], loc='best')
        plt.savefig('../scripts/figure/wrench.png')
        plt.suptitle('wrench')

        plt.figure()
        plt.subplot(331)
        plt.plot(self.time_step, plot_t[0])
        plt.ylim((0, 10))
        plt.legend(labels=['t1'], loc='best')
        plt.subplot(332)
        plt.plot(self.time_step, plot_t[1])
        plt.ylim((0, 10))
        plt.legend(labels=['t2'], loc='best')
        plt.subplot(333)
        plt.plot(self.time_step, plot_t[2])
        plt.ylim((0, 10))
        plt.legend(labels=['t3'], loc='best')
        plt.subplot(334)
        plt.plot(self.time_step, plot_t[3])
        plt.ylim((0, 10))
        plt.legend(labels=['t4'], loc='best')
        plt.subplot(335)
        plt.plot(self.time_step, plot_t[4])
        plt.ylim((0, 10))
        plt.legend(labels=['t5'], loc='best')
        plt.subplot(336)
        plt.plot(self.time_step, plot_t[5])
        plt.ylim((0, 10))
        plt.legend(labels=['t6'], loc='best')
        plt.subplot(337)
        plt.plot(self.time_step, plot_t[6])
        plt.ylim((0, 10))
        plt.legend(labels=['t7'], loc='best')
        plt.savefig('../scripts/figure/tensions.png')
        plt.suptitle('tensions')

        plt.figure()
        plt.subplot(311)
        plt.plot(self.time_step, plot_pos[0])
        plt.legend(labels=['x'], loc='best')
        plt.subplot(312)
        plt.plot(self.time_step, plot_pos[1])
        plt.legend(labels=['y'], loc='best')
        plt.subplot(313)
        plt.plot(self.time_step, plot_pos[2])
        plt.legend(labels=['z'], loc='best')
        plt.savefig('../scripts/figure/platform_pos.png')
        plt.suptitle('platform_pos')
        plt.show()


class apmPID:
    def __init__(self, target):
        # position
        x_p, x_i, x_d = 0.1, 0, 0.05
        y_p, y_i, y_d = 0.1, 0, 0.05
        z_p, z_i, z_d = 0.15, 0, 0.05

        # 倍数
        x, y, z = 2.5, 2.5, 1.5
        p = 1.0
        self.control_x = PID(np.asarray([x_p, x_i, x_d]) * x * p, target[0], upper=100,
                             lower=-100)  # control position x
        self.control_y = PID(np.asarray([y_p, y_i, y_d]) * y * p, target[1], upper=100,
                             lower=-100)  # control position y
        self.control_z = PID(np.asarray([z_p, z_i, z_d]) * z * p, target[2], upper=100,
                             lower=-100)  # control position z

    def cal_actions(self, state):
        '''
        :param state: 目标位姿
        :return: 力+力矩
        '''
        u1 = self.control_x.cal_output(state[0])
        u2 = self.control_y.cal_output(state[1])
        u3 = self.control_z.cal_output(state[2])
        list = [u1, u2, u3]
        return list


if __name__ == "__main__":
    env = CDPR()
    env.run()
    # env.run_original()
