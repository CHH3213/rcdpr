# -*-coding:utf-8-*-
'''
测试插件的力是否正确
'''
from __future__ import print_function, absolute_import, division
import sys
import rospy
from std_msgs.msg import Float32MultiArray
import os
import gym
import math
import scipy.io as sio
from gazebo_msgs.srv import *
import numpy as np
import matplotlib.pyplot as plt
import time
sys.path.append('/home/firefly/chh_ws/src/rcdpr/scripts')
from control_drone import Drone
from control_car import Omni_car
from gazebo_reset import Gazebo_reset
from geometry_msgs.msg import Pose, Twist


class CDPR(gym.Env):
    def __init__(self):
        # # 物理属性
        self.payload_gravity = 2.5
        self.drone_gravity = 18.88
        self.dt = 0.1  ## 仿真频率
        self.save_data_dir = "./scripts/data"
        if not os.path.exists(self.save_data_dir):
            os.makedirs(self.save_data_dir)


        self.force_publisher0 = rospy.Publisher('/rcdpr_force', Float32MultiArray, queue_size=1)

        ### Initiate ROS node
        print('-- Connecting to mavros')
        rospy.init_node('cdpr_mavros',anonymous=True)
        print('connected')
        # self.Drone1 = Drone('drone1')
        # self.Drone2 = Drone('drone2')
        # self.Drone3 = Drone('drone3')
        # self.Drone4 = Drone('drone4')
        self.Car0 = Omni_car('omni_car_0')
        self.Car1 = Omni_car('omni_car_1')
        self.Car2 = Omni_car('omni_car_2')
        self.gazebo_reset = Gazebo_reset()


        self.cable_number = 7  # 绳子数量
        self.cable_length = np.array([4.5 for _ in range(self.cable_number)])

        # value function 需要保存的输入输出数据
        self.input = []
        self.output = []

    def run(self):
        '''
        运行代价函数收集数据
        :return:
        '''

        self.gazebo_reset.resetPayloadState()
        time.sleep(0.1)
        self.gazebo_reset.reset_Car_State([self.Car0,self.Car1,self.Car2])
        time.sleep(0.1)

        multiarray0 = Float32MultiArray()
        self.start = time.time()
        hz = 1000
        hz1=1000
        r = rospy.Rate(hz1)
        count_draw = 1
        # 初始平台位置
        self.dist_x = 0
        self.dist_y = 0
        self.dist_z = 3

        # 圆形轨迹圆心和半径长
        radius = 2
        c0 = [2, 0]
        theta = 0
        '''
        无人机解锁并起飞
        '''
        # self.Drone1.set_mode('GUIDED')
        # self.Drone2.set_mode('GUIDED')
        # self.Drone3.set_mode('GUIDED')
        # self.Drone4.set_mode('GUIDED')
        # self.Drone1.set_arm(True)
        # self.Drone2.set_arm(True)
        # self.Drone3.set_arm(True)
        # self.Drone4.set_arm(True)
        # self.Drone1.takeoff(8.0)
        # self.Drone2.takeoff(8.0)
        # self.Drone3.takeoff(8.0)
        # self.Drone4.takeoff(8.0)
        # rospy.sleep(5)
        # self.Drone1.goto_xyz(0.87, 1.5, 8.0)
        # self.Drone2.goto_xyz(-1.74, 0, 8.0)
        # self.Drone3.goto_xyz(0.87, -1.5, 8.0)
        # self.Drone4.goto_xyz(2, 0, 8.0)
        # rospy.sleep(6)

        dist_x, dist_y = 0,0
        self.gazebo_reset.set_payload_state()  # 将payload初始化到(0,0,3)
        while not rospy.is_shutdown():
            # 无人机的机体坐标刚好和世界坐标反过来了
            start = time.time()
            # self.Drone1.goto_xyz(1.0, 0.0, 8.0, 0, 0, 0)
            # self.Drone2.goto_xyz(-self.dist_y,self.dist_x,8.0)
            # self.Drone3.goto_xyz(-self.dist_y,self.dist_x,8.0)
            # self.Drone4.goto_xyz(-2,-2,8.0)

            # 测试小车走圆形轨迹
            theta += 2 * np.pi / (10000)
            temp = [dist_x, dist_y]
            dist_x = 1.1 * (radius * np.cos(np.pi - theta) + c0[0])  # 小车误差修正
            dist_y = 1.1 * radius * np.sin(np.pi - theta) + c0[1]
            temp_pos = [dist_x - temp[0], dist_y - temp[1]]  # 质点一步要走的距离
            print('temp_pos', temp_pos)
            # 小车运动
            self.Car2.goto_car_xyz(temp_pos, dt = self.dt)  
            # vel = Twist()
            # vel.linear.x = 0
            # vel.linear.y = 1
            # self.Car2.car_vel.publish(vel)
            # time.sleep(10)
            # drone--force
            force_payload2drone1 = 0.0
            force_payload2drone2 = 0.0
            force_payload2drone3 = 0.0
            force_payload2drone4 = 0.98
            # 小车共用一个话题
            force_logger0 = 0.0
            force_logger1 = 0.0
            force_logger2 = 0.0
            multiarray0.data = [force_logger0 * 1000/hz, force_logger1 * 1000/hz, force_logger2 * 1000/hz, force_payload2drone1 *
                                1000/hz, force_payload2drone2 * 1000/hz, force_payload2drone3 * 1000/hz, force_payload2drone4 * 1000/hz]

            print('hahah')
            self.force_publisher0.publish(multiarray0)
            print('load plugin success')
            r.sleep()
            count_draw += 1
            self.dt = time.time()-start
            hz = 1/(time.time()-start)
            # print('self.dt',self.dt)
            print(time.time()-start)
            if count_draw == 10000:
                print('break')
                break

        # self.Drone1.set_mode('RTL')
        # self.Drone2.set_mode('RTL')
        # self.Drone3.set_mode('RTL')
        # self.Drone4.set_mode('RTL')

        self.gazebo_reset.reset_Car_State([self.Car0,self.Car1,self.Car2])
        self.gazebo_reset.resetPayloadState()

    def quaternion_to_euler(self, x, y, z, w):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))
        
        t2 = +2.0 * (w * y - z * x)
        # t2 = +1.0 if t2 > +1.0 else t2
        # t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        # 使用 tf 库
        # import tf
        # (X, Y, Z) = tf.transformations.euler_from_quaternion([x, y, z, w])
        return X, Y, Z

    def goto_car_xyz(self, vel_pub, odom, target_pos):
        '''
        小车运动
        :param vel_pub:  速度话题
        :param target_pos:  目标位置
        :param current_attitude: 当前姿态，欧拉角
        :return:
        '''

        x,y,z = self.quaternion_to_euler(odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)
        print('pose',z)
        vel = Twist()
        vx = (target_pos[0])/self.dt
        vy = (target_pos[1]) / self.dt
        vel.linear.x = vx*np.cos(z)+vy*np.sin(z)
        vel.linear.y = -vx*np.sin(z)+vy*np.cos(z)
        vel_pub.publish(vel)
if __name__ == "__main__":
    env = CDPR()
    env.run()
    # env.run_original()
