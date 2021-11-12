# -*-coding:utf-8-*-
'''
无人机发布订阅命令
'''
from __future__ import print_function, absolute_import, division
# ROS packages required
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from mavros_msgs.msg import ActuatorControl
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.msg import PositionTarget
from mavros_msgs.msg import GlobalPositionTarget
from mavros_msgs.msg import Thrust
from mavros_msgs.msg import RCOut
from mavros_msgs.msg import RCIn
from mavros_msgs.msg import OverrideRCIn
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import SetModeRequest
from mavros_msgs.srv import SetModeResponse
from mavros_msgs.srv import CommandTOL
from mavros_msgs.srv import CommandBoolRequest
from mavros_msgs.srv import CommandBoolResponse
from mavros_msgs.srv import StreamRate, StreamRateRequest

from math import *
from sensor_msgs.msg import Imu
from sensor_msgs.msg import BatteryState

from gazebo_msgs.srv import *
from gazebo_msgs.srv import SetModelStateRequest
from geometry_msgs.msg import Pose, Twist
from rotate_calculation import Rotate

class Drone:
    def __init__(self, name):

        # # parameters
        # self.log_dir = "../log"
        # if not os.path.exists(self.log_dir):
        #     os.makedirs(self.log_dir)

        # states
        self.state = State()
        self.battery = BatteryState()
        self.rc_states = RCOut()
        self.local_position = PoseStamped()

        # clients
        rospy.loginfo("waiting for ROS services")
        rospy.wait_for_service(name + '/' + 'mavros/cmd/arming')  # make sure that your service is available
        rospy.wait_for_service(name + '/' + 'mavros/set_mode')
        rospy.wait_for_service(name + '/' + 'mavros/cmd/takeoff')
        self.takeoff_client = rospy.ServiceProxy(name + '/' + 'mavros/cmd/takeoff', CommandTOL)
        self.arming_client = rospy.ServiceProxy(name + '/' + 'mavros/cmd/arming',CommandBool)  # name and srv type (rosservice info <srv_name>)
        self.set_mode_client = rospy.ServiceProxy(name + '/' + 'mavros/set_mode', SetMode)

        # pubs
        self.local_pos_pub = rospy.Publisher(name + '/' + 'mavros/setpoint_position/local', PoseStamped, queue_size=1)
        self.rc_override = rospy.Publisher(name + '/' + "mavros/rc/override", OverrideRCIn, queue_size=1)
        # subs
        self.state_sub = rospy.Subscriber(name + '/' + 'mavros/state', State, self.state_callback,queue_size=1,buff_size=52428800)
        self.rc_out_sub = rospy.Subscriber(name + '/' + "mavros/rc/out", RCOut, self.rc_out_cb, queue_size=1,buff_size=52428800)
        self.local_pos_sub = rospy.Subscriber(name + '/' +"mavros/local_position/pose", PoseStamped, self.lp_cb, queue_size=1,buff_size=52428800)

        self.Rotate = Rotate()

    # callback funcs
    def state_callback(self, data):
        # change notice
        if self.state.armed != data.armed:
            rospy.loginfo("armed state changed from {0} to {1}".format(
                self.state.armed, data.armed))
        if self.state.connected != data.connected:
            rospy.loginfo("connected changed from {0} to {1}".format(
                self.state.connected, data.connected))
        if self.state.mode != data.mode:
            rospy.loginfo("mode changed from {0} to {1}".format(
                self.state.mode, data.mode))
        self.state = data

    def rc_out_cb(self, data):
        self.rc_states = data
    def lp_cb(self,data):
        self.local_position = data
    # helper methods

    def get_posByMavros(self):
        '''
        使用mavros话题获取无人机状态
        :return: 返回无人机的位置和欧拉角姿态
        '''
        pos_drone = [self.local_position.pose.position.x,self.local_position.pose.position.y,self.local_position.pose.position.z]
        ori_drone = self.Rotate.quaternion_to_euler(self.local_position.pose.orientation.x,self.local_position.pose.orientation.y,self.local_position.pose.orientation.z,self.local_position.pose.orientation.w)
        return [pos_drone,ori_drone]

    def set_arm(self, arm):
        """arm: True to arm or False to disarm"""
        rospy.loginfo("setting FCU arm: {0}".format(arm))
        loop_freq = 2  # 2Hz
        rate = rospy.Rate(loop_freq)
        counter = 0
        while self.state.armed != arm:
            rospy.logerr("failed to send arm command")
            self.arming_client(arm)
            counter = counter + 1
            rate.sleep()
        rospy.loginfo("set arm success | seconds: {0} ".format(counter / loop_freq))

    def set_mode(self, mode):
        """mode: APM mode string"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        loop_freq = 2  # 2Hz
        rate = rospy.Rate(loop_freq)
        counter = 0
        while self.state.mode != mode:
            print(self.state.mode)
            rospy.logerr("failed to send set mode command")
            self.set_mode_client(0, mode)
            counter = counter + 1
            rate.sleep()
        rospy.loginfo("set mode success | seconds: {0} ".format(counter / loop_freq))

    def takeoff(self, altitude):
        if self.state.armed == True:
            self.takeoff_client(altitude=altitude)

    def goto_xyz(self,x, y, z):
        """
        Set the given pose as a the next setpoint by sending
        a SET_POSITION_TARGET_LOCAL_NED message. The copter must
        be in GUIDED mode for this to work.
        :param x:
        :param y:
        :param z:
        :return:
        """
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time()
        pose_stamped.pose = pose
        self.local_pos_pub.publish(pose_stamped)

    def goto_xyz_rpy(self, x, y, z, ro=0, pi=0, ya=0):
        '''
        Set the given pose as a the next setpoint by sending
        a SET_POSITION_TARGET_LOCAL_NED message. The copter must
        be in GUIDED mode for this to work.
        :param x:
        :param y:
        :param z:
        :param ro:
        :param pi:
        :param ya:
        :return:
        '''
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        import tf
        quat = tf.transformations.quaternion_from_euler(ro, pi, ya)

        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time()
        pose_stamped.pose = pose
        self.local_pos_pub.publish(pose_stamped)

if __name__ == "__main__":

    # 启动mavros setpoint，查看local pos高度校正效果

    #
    rospy.init_node("multi_drones", anonymous=True)

    drone1 = Drone('drone1')    	# 最好roslaunch传参，保证ns与mavros相同
    drone2 = Drone('drone2')

    # drone1.set_mode('STABILIZE')	# 遥控器油门最低，否则不能解锁
    # drone2.set_mode('STABILIZE')
    # drone1.set_arm(True)
    # drone2.set_arm(True)

    drone1.set_mode('GUIDED')
    drone2.set_mode('GUIDED')
    drone1.set_arm(True)
    drone2.set_arm(True)

    rospy.sleep(5)

    drone1.takeoff(1)
    drone2.takeoff(1)
    rospy.sleep(15)

    drone1.goto_xyz(0.3, 0.3, 1)
    drone2.goto_xyz(0.3, 0.3, 1)
    rospy.sleep(15)

    drone1.goto_xyz(0, 0, 1)
    drone2.goto_xyz(0, 0, 1)
    rospy.sleep(15)

    drone1.set_mode('LAND')
    drone2.set_mode('LAND')
    rospy.sleep(10)
    drone1.set_arm(False)
    drone2.set_arm(False)
