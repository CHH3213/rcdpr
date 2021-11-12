# -*-coding:utf-8-*-
'''
小车控制：发布、订阅 命令
'''
from __future__ import print_function, absolute_import, division
# ROS packages required
import rospy

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3

from math import *
from sensor_msgs.msg import Imu
from sensor_msgs.msg import BatteryState

from std_msgs.msg import Header
from std_msgs.msg import Float64
from std_srvs.srv import Empty, EmptyRequest
from gazebo_msgs.srv import SetModelState, GetModelState  # 设置模型状态、得到模型状态
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import *
from gazebo_msgs.srv import SetModelStateRequest
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import  Odometry
import numpy as np
from rotate_calculation import Rotate

class Omni_car:
    def __init__(self, name):

        # # parameters
        # self.log_dir = "../log"
        # if not os.path.exists(self.log_dir):
        #     os.makedirs(self.log_dir)

        # states
        self.model_states = ModelStates()
        self.odom = Odometry()
        self.name = name

        # clients
        rospy.loginfo("waiting for ROS services")

        # pubs
        self.car_vel = rospy.Publisher(name + '/' + 'cmd_vel', Twist, queue_size=1)
        # subs
        self.car0_pose = rospy.Subscriber(name + '/' + "odom", Odometry, self.pose_cb, queue_size=1, buff_size=52428800)
        self.rotate = Rotate()

    # callback funcs
    def pose_cb(self, data):
        self.odom = data
    # helper methods
    def goto_car_xyz(self, target_pos, current_attitude=0,dt=0.1):
        '''
        小车运动
        :param target_pos:  要走的目标位置
        :param current_attitude: 当前姿态，欧拉角
        :param dt: 一个step运行的时间
        :return:
        '''

        x,y,z = self.rotate.quaternion_to_euler(self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w)
        # print('pose z:{}'.format(z))
        # print('current_attitude',current_attitude[2])  # 绕自身z周旋转的角度
        vel = Twist()
        # print('dt', dt)
        vx = (target_pos[0])/dt
        vy = (target_pos[1]) / dt
        # print('vx,vy', vx, vy)
        # z = current_attitude[2]
        vel.linear.x = vx*np.cos(z)+vy*np.sin(z)
        vel.linear.y = -vx*np.sin(z)+vy*np.cos(z)

        # vel.linear.x = vx
        # vel.linear.y = vy

        # print('vel',vel.linear.x,vel.linear.y)
        self.car_vel.publish(vel)


if __name__ == "__main__":
    pass


