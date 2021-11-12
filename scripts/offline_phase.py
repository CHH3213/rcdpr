# -*-coding:utf-8-*-
'''
4架无人机，3辆小车
通过代价函数获取数据，离线训练神经网络
'''
from __future__ import print_function, absolute_import, division

import rospy
from std_msgs.msg import Float32MultiArray
import os
from PIDClass import PID
import gym
import scipy.io as sio
from gazebo_msgs.srv import *
import numpy as np
import matplotlib.pyplot as plt
from control_drone import Drone
from control_car import Omni_car
from gazebo_reset import Gazebo_reset
from rotate_calculation import Rotate
from value_function import Value_function
from Jacobi import jacobi
from module_test import replay_buffer
from optimizer import *
from pidController import *
import time
import copy
np.set_printoptions(suppress=True)  # 不以科学计数法输出



class CDPR(gym.Env):
    def __init__(self):
        # # 物理属性
        self.payload_gravity = 0.98
        self.drone_gravity = 18.88
        self.dt = 0.1  ## 仿真频率
        self.save_data_dir = "../scripts/data"
        if not os.path.exists(self.save_data_dir):
            os.makedirs(self.save_data_dir)

        self.force_publisher0 = rospy.Publisher('/rcdpr_force', Float32MultiArray, queue_size=1)

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
        self.Rot = Rotate()
        # TODO:参数的具体定义
        self.m = self.payload_gravity / 9.8
        self.d_min = 0.1
        self.mu_t, self.mu_r, self.mu_d, self.mu_a, self.mu_l = 1.0, 1.0, 0.02, 0.2, 0.2
        self.t_min = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.t_max = np.array([5, 5, 5, 5, 5, 5, 5])
        self.target_length_list = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
        self.value_function = Value_function(self.m, d_min=self.d_min, mu_t=self.mu_t,
                                             mu_r=self.mu_r, mu_d=self.mu_d, mu_a=self.mu_a,
                                             t_min=self.t_min, t_max=self.t_max, mu_l=self.mu_l,
                                             target_length_list=self.target_length_list)
        # 末端执行器固定点在其自身坐标系中的位置uav1~4,ugv1~3
        self.point_end_effector = np.array([[0,0,0] for _ in range(7)])
        self.cable_number = 7  # 绳子数量
        self.cable_length = np.array([5 for _ in range(self.cable_number)])
        self.cable_one_side = np.empty((7, 3))  # 7个固定点，每个点是 3惟的

        # value function 需要保存的输入输出数据
        self.input = []
        self.input_platform = []
        self.output = []
        self.r_t = []
        self.r_r = []
        # value输入保存
        self.CABLE_ONE_SIDE = []
        self.CABLE_OTHER_SIDE = []
        self.ROTATION_CENTER = []
        self.POSE_0 = []

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
        self.drone1_pos = []
        self.drone2_pos = []
        self.drone3_pos = []
        self.drone4_pos = []
        self.platform_quat = []
        self.platform_euler = []


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

        hz = 100
        r = rospy.Rate(hz)
        count_draw = 1
        # 初始平台位置
        self.dist_x = 0
        self.dist_y = 0
        self.dist_z = 4.0
        dist_x = 0  # 小车误差修正
        dist_y = 0
        # 圆形轨迹圆心和半径长
        radius = 0.2
        c0 = [0.2, 0]
        theta = 0
        total_steps = 5000
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
        # rospy.sleep(5)
        self.Drone1.goto_xyz(0.87, 1.5, 8.0)
        self.Drone2.goto_xyz(-1.74, 0, 8.0)
        self.Drone3.goto_xyz(0.87, -1.5, 8.0)
        self.Drone4.goto_xyz(2, 0, 8.0)
        rospy.sleep(6)
        self.count = 0
        # TODO:payload初始化
        self.gazebo_reset.set_payload_state()  # 将payload初始化到(0,0,3)
        pid = apmPID()
        # [self.payload_position, self.payload_attitude, self.payload_pose0, self.payload_linear, _] = self.get_state(
        #     self.gazebo_reset.model_states, 'payload')
        # print('============pos====', self.payload_position)
        time.sleep(0.015)
        while not rospy.is_shutdown():
            # 画圆形轨迹 (x-x0)^2+(y-y0)^2=r^2  <==> 令圆心为（2，0），半径为2，==》x = 2cos(pi-theta)+2,  y=sin(pi-theta),在1000步时刚好画完一个圆
            start = time.time()
            '''=====================轨迹=========================='''
            if count_draw > 1000:
                # self.dist_z += 1.0/(total_steps-1000)  # z轴向上运动1.5m
                theta += 2 * np.pi / (total_steps - 1000)
                # print('theat',theta)
                temp = [dist_x, dist_y]
                self.dist_x = radius * np.cos(np.pi - theta) + c0[0]
                self.dist_y = radius * np.sin(np.pi - theta) + c0[1]
                # dist_x = 1.1 * (radius * np.cos(np.pi - theta) + c0[0])  # 小车误差修正
                # dist_y = 1.1 * radius * np.sin(np.pi - theta) + c0[1]
                # print('temp',temp)
                # print('dist',dist_x,dist_y)
                # temp_pos = [dist_x - temp[0], dist_y - temp[1]]  # 质点一步要走的距离
                # print('distance', temp_pos)
            #     # 小车运动
            #     self.Car0.goto_car_xyz(temp_pos, self.logger0_attitude, self.dt)
            #     # time.sleep(0.0005)
            #     self.Car1.goto_car_xyz(temp_pos, self.logger1_attitude, self.dt)
            #     # time.sleep(0.0005)
            #     self.Car2.goto_car_xyz(temp_pos, self.logger2_attitude, self.dt)
            '''==================================='''
            # 无人机的机体坐标刚好和世界坐标反过来了
            
            # self.Drone1.goto_xyz(-self.dist_y * 2 + 0.87, self.dist_x * 2 + 1.5, 8.0)
            # # time.sleep(0.0005)
            # self.Drone2.goto_xyz(-self.dist_y * 2 - 1.74, self.dist_x * 2, 8.0)
            # # time.sleep(0.0005)
            # self.Drone3.goto_xyz(-self.dist_y * 2 + 0.87, self.dist_x * 2 - 1.5, 8.0)
            # # time.sleep(0.0005)
            # self.Drone4.goto_xyz(-self.dist_y * 2 + 2, self.dist_x * 2, 8.0)
            # # time.sleep(0.005)

            '''######### 质点位置目标  ###########'''
            # ***********************************
            target = [self.dist_x, self.dist_y, self.dist_z]
            # print('payload_target', target)
            pid.set_target(target)
            # uav:1~4,ugv:1~3
            # TODO:让PID稳定的更久一点
            cable_tensions = self.compute_cable_tensions(pid)

            if cable_tensions is None:
                print('step ', count_draw)
                break
            ########################################################
            # drone--force
            # TODO:这里不该再有cable_tension的非负限制
            # for i in range(len(cable_tensions)):
            #     if cable_tensions[i] < -10:
            #         cable_tensions[i] = -10
            # cable_tensions = np.where(cable_tensions > 0, cable_tensions, 0)

            force_payload2drone1 = cable_tensions[0]
            force_payload2drone2 = cable_tensions[1]
            force_payload2drone3 = cable_tensions[2]
            force_payload2drone4 = cable_tensions[3]
            # 小车共用一个话题
            force_logger0 = cable_tensions[4]
            force_logger1 = cable_tensions[5]
            force_logger2 = cable_tensions[6]
            if count_draw > 2:
                hz = 1.0 / (time.time() - s)
            coefficience = 1000 / hz
            # coefficience = 10
            # print('coeff ', coefficience)
            multiarray0.data = [force_logger0 * coefficience, force_logger1 * coefficience, force_logger2 * coefficience,
                                force_payload2drone1 * coefficience, force_payload2drone2 * coefficience,
                                force_payload2drone3 * coefficience, force_payload2drone4 * coefficience]
            # 无人机共用一个话题
            # print('pos', self.drone1_position[2])
            s = time.time()
            if self.drone1_position[2] > 0 and self.drone2_position[2] > 0 and self.drone3_position[2] > 0 and \
                    self.drone4_position[2] > 0:
                self.force_publisher0.publish(multiarray0)

            
            '''生成代价函数数据---耗时'''
            # self.generate_train_data()
            '''-----------------------------------'''
            r.sleep()
            self.dt = time.time() - start
            print('compute_time', time.time() - start)
            count_draw += 1
            self.time_step.append(count_draw)
            if count_draw == total_steps:
                # self.plot_curve()
                sio.savemat(self.save_data_dir + '/' + 'static_02_trajectry_{}.mat'.format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())),
                            dict(steps=self.time_step, drone1_pos=self.drone1_pos, drone2_pos=self.drone2_pos, drone3_pos=self.drone3_pos,
                                 drone4_pos=self.drone4_pos, platform_pos=self.platform_pos,
                                  platform_quat=self.platform_quat, platform_euler=self.platform_euler))

                # sio.savemat(self.save_data_dir + '/'+'value_data_test_{}.mat'.format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())),
                #             dict(input_platform=self.input_platform, input_pos=self.input, output_value=self.output,
                #                  platform_pos=self.platform_pos, r_t=self.r_t, r_r=self.r_r))

                sio.savemat(self.save_data_dir + '/'+'static_02_value_input_{}.mat'.format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())),
                            dict(CABLE_ONE_SIDE=self.CABLE_ONE_SIDE, CABLE_OTHER_SIDE=self.CABLE_OTHER_SIDE, POSE_0=self.POSE_0,
                                 ROTATION_CENTER=self.ROTATION_CENTER, point_end_effector = self.point_end_effector))
                sio.savemat(self.save_data_dir + '/static_02_force_input_{}.mat'.format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())),
                            dict(wrench=self.wrench, tensions=self.t))
                # sio.savemat(self.save_data_dir + '/plot_data.mat',
                #             dict(steps=self.time_step, platform_pos=self.platform_pos, wrench=self.wrench,
                #                  tensions=self.t))
                break
            # hz = 1.0 / (time.time() - start)
        
        # self.Drone1.set_mode('RTL')
        # self.Drone2.set_mode('RTL')
        # self.Drone3.set_mode('RTL')
        # self.Drone4.set_mode('RTL')

        self.gazebo_reset.reset_Car_State([self.Car0, self.Car1, self.Car2])
        self.gazebo_reset.resetPayloadState()
        # value function 需要保存的输入输出数据
        self.input = []
        self.input_platform = []
        self.output = []
        self.r_t = []
        self.r_r = []
        # 画图数据
        self.wrench = []
        self.t = []
        self.time_step = []
        self.platform_pos = []
        self.drone1_pos = []
        self.drone2_pos = []
        self.drone3_pos = []
        self.drone4_pos = []
        self.platform_quat = []
        self.platform_euler = []
        # value输入保存
        self.CABLE_ONE_SIDE = []
        self.CABLE_OTHER_SIDE = []
        self.ROTATION_CENTER = []
        self.POSE_0 = []
        print('结束！')

    def compute_cable_tensions(self, pid):
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
        # TODO:PID修改
        target_force_wrench = pid.cal_actions(self.payload_position)
        # print('pos', self.payload_position)
        compensate_wrench = np.array([0, 0, self.payload_gravity, 0, 0, 0])  # 补偿负载重力
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

        # TODO: 新的雅可比---不需要力矩
        self.Jac, cable_tensions, force_wrench = jacobi(self.points_base, point_end_effector, pose_0, rotation_center,
                                                        target_force_wrench)
        '''save plot data'''
        self.wrench.append(target_force_wrench)
        self.t.append(cable_tensions)
        self.platform_pos.append(self.payload_position)
        self.drone1_pos.append(self.drone1_position)
        self.drone2_pos.append(self.drone2_position)
        self.drone3_pos.append(self.drone3_position)
        self.drone4_pos.append(self.drone4_position)
        self.platform_quat.append(self.payload_pose0)
        self.platform_euler.append(self.payload_attitude)
        # value input save
        self.CABLE_ONE_SIDE.append(self.cable_one_side)
        self.CABLE_OTHER_SIDE.append(self.points_base)
        self.ROTATION_CENTER.append(rotation_center)  # platform position
        self.POSE_0.append(pose_0)  # platform pose

        # print('cable tensions: ',cable_tensions)
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
            if self.count > 100:
                return None
        #     print('cable_tensions', cable_tensions)
        #     cable_tensions = self.replay_buff.sample(1)
        # else:
        #     self.replay_buff.push(cable_tensions)
        # # 测试是否还有小于0的数
        # is_empty = np.where(cable_tensions < -1e-15)
        # is_empty = np.array(is_empty)
        # if is_empty.size > 0:
        #     print('cable_tensions', cable_tensions)


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
        return copy.deepcopy([model_position, model_attitude, model_pose0, model_linear, model_angular])

    def generate_train_data(self):
        '''
        使用8+6维度作为输入
        生成代价函数
        :return:
        '''
        # 代价函数处理
        self.value_function.set_jacobian_param(self.point_end_effector,self.payload_pose0,self.payload_position)
        cables_other_side = self.points_base  # 绳子ling一端--agent固定点
        # print(np.shape(self.cable_one_side))  #7*3
        # cable_lines = np.array([np.array([self.cable_one_side[i], cables_other_side[i]]) for i in range(len(self.cable_one_side))])
        # print('cable', cable_lines)
        # print(len(cable_lines))
        cost_cable_interference = self.value_function.cost_cable_interference(self.cable_one_side,
                                                                              cables_other_side)
        feasible_sets = self.cable_length
        cost_feasible_points = self.value_function.cost_feasible_points(self.cable_one_side, cables_other_side,feasible_sets, None)
        

        r_r_AW = self.value_function.r_r_AW(self.cable_one_side,cables_other_side,self.payload_position)
        r_t_AW = self.value_function.r_t_AW(self.cable_one_side,cables_other_side)
        cost_wrench = r_r_AW + r_t_AW
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
        self.r_r.append(-r_r_AW/self.mu_r)
        self.r_t.append(-r_t_AW/self.mu_t)

    def plot_curve(self):
        plot_wrench = np.array(self.wrench).T
        plot_t = np.array(self.t).T
        plot_pos = np.array(self.platform_pos).T
        drone1, drone2, drone3, drone4 = np.array(self.drone1_pos).T, np.array(self.drone2_pos).T, np.array(self.drone3_pos).T, np.array(self.drone4_pos).T


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
        plt.suptitle('wrench')
        plt.savefig('../scripts/figure/wrench.png')

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
        plt.suptitle('tensions')
        plt.savefig('../scripts/figure/tensions.png')

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
        plt.suptitle('platform_pos')
        plt.savefig('../scripts/figure/platform_pos.png')

        plt.figure()
        plt.subplot(311)
        plt.plot(self.time_step,drone1[0])
        plt.plot(self.time_step,drone2[0])
        plt.plot(self.time_step,drone3[0])
        plt.plot(self.time_step,drone4[0])
        plt.legend(labels=['drone1-x', 'drone2-x', 'drone3-x', 'drone4-x'], loc='best')

        plt.subplot(312)
        plt.plot(self.time_step, drone1[1])
        plt.plot(self.time_step, drone2[1])
        plt.plot(self.time_step, drone3[1])
        plt.plot(self.time_step, drone4[1])
        plt.legend(labels=['drone1-y', 'drone2-y', 'drone3-y', 'drone4-y'], loc='best')

        plt.subplot(313)
        plt.plot(self.time_step, drone1[2])
        plt.plot(self.time_step, drone2[2])
        plt.plot(self.time_step, drone3[2])
        plt.plot(self.time_step, drone4[2])
        plt.legend(labels=['drone1-z', 'drone2-z', 'drone3-z', 'drone4-z'], loc='best')
        plt.suptitle('drone_pos')
        plt.savefig('../scripts/figure/drone_pos.png')

        plt.figure()
        plt.plot(plot_pos[0, 1000:], plot_pos[1, 1000:])
        plt.plot(drone1[0, 1000:], drone1[1, 1000:])
        plt.plot(drone2[0,1000:], drone2[1,1000:])
        plt.plot(drone3[0,1000:], drone3[1,1000:])
        plt.plot(drone4[0,1000:], drone4[1,1000:])
        plt.suptitle('trajectory')
        plt.savefig('../scripts/figure/trajectory.png')
        plt.show()



class apmPID:
    def __init__(self, target = np.array([0,0,0])):
        # position 0.2: xp,yp = 0.05, 0.1:xp,yp = 0.1
        x_p, x_i, x_d = 0.05, 0.00, 0.01
        y_p, y_i, y_d = 0.05, 0.00, 0.01
        z_p, z_i, z_d = 0.09, 0.000, 0.09
        # 倍数
        x, y, z = 1, 1, 1.25
        p = 2.5 * 1.5
    


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
        # print('current state: ',state)
        # print('pid output: ',list)
        return list
    
    def set_target(self, target):
        self.control_x.set_target(target=target[0])
        self.control_y.set_target(target=target[1])
        self.control_z.set_target(target=target[2])
        # print('target: ', target)



if __name__ == "__main__":
    env = CDPR()
    for i in range(30):
        print('Round:{}'.format(i+1))
        env.run()
    # env.run_original()
