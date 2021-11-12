# -*-coding:utf-8-*-
'''
通过gazebo来reset状态
'''
from __future__ import print_function, absolute_import, division
import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState  # 设置模型状态、得到模型状态
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import *
from gazebo_msgs.srv import SetModelStateRequest
from geometry_msgs.msg import Pose, Twist
import time


class Gazebo_reset:
    def __init__(self):
        # gazebo topic
        self.modestate_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_cb)
        # gazebo服务
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)  # 指定服务名来调用服务
        self.reset_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

    def set_payload_state(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            my_model = SetModelStateRequest()
            my_model.model_state.model_name = 'payload'
            my_model.model_state.pose.position.x = 0.0
            my_model.model_state.pose.position.y = 0.0
            my_model.model_state.pose.position.z = 4.0
            my_model.model_state.pose.orientation.x = 0
            my_model.model_state.pose.orientation.y = 0
            my_model.model_state.pose.orientation.z = 0
            my_model.model_state.pose.orientation.w = 1
            my_model.model_state.twist.linear.x = 0.0
            my_model.model_state.twist.linear.y = 0.0
            my_model.model_state.twist.linear.z = 0.0
            my_model.model_state.twist.angular.x = 0.0
            my_model.model_state.twist.angular.y = 0.0
            my_model.model_state.twist.angular.z = 0.0
            my_model.model_state.reference_frame = "world"
            self.reset_proxy(my_model)
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/set_modelState service call failed")

    def resetPayloadState(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            my_model = SetModelStateRequest()
            my_model.model_state.model_name = 'payload'
            my_model.model_state.pose.position.x = 0.0
            my_model.model_state.pose.position.y = 0.0
            # my_model.model_state.pose.position.z = 0.0
            my_model.model_state.pose.orientation.x = 0
            my_model.model_state.pose.orientation.y = 0
            my_model.model_state.pose.orientation.z = 0
            my_model.model_state.pose.orientation.w = 1
            my_model.model_state.twist.linear.x = 0.0
            my_model.model_state.twist.linear.y = 0.0
            my_model.model_state.twist.linear.z = 0.0
            my_model.model_state.twist.angular.x = 0.0
            my_model.model_state.twist.angular.y = 0.0
            my_model.model_state.twist.angular.z = 0.0
            my_model.model_state.reference_frame = "world"
            self.reset_proxy(my_model)
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_modelState service call failed")

    def reset_Car_State(self, omni_cars):
        '''

        :param omni_cars: 列表， 元素为car类
        :return:
        '''
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            vel = Twist()
            vel.linear.x = 0
            vel.linear.y = 0
            vel.linear.z = 0
            vel.angular.x = 0
            vel.angular.y = 0
            vel.angular.z = 0
            my_model = SetModelStateRequest()
            for omni_car in omni_cars:
                omni_car.car_vel.publish(vel)
                my_model.model_state.model_name = omni_car.name
                if omni_car.name == 'omni_car_0':
                    my_model.model_state.pose.position.x = 2.5
                    my_model.model_state.pose.position.y = 1.45

                elif omni_car.name == 'omni_car_1':
                    my_model.model_state.pose.position.x = -2.5
                    my_model.model_state.pose.position.y = 1.45

                elif omni_car.name == 'omni_car_2':
                    my_model.model_state.pose.position.x = 0
                    my_model.model_state.pose.position.y = -2.9
                else:
                    assert 'car‘s name may be properly wrong.'

                my_model.model_state.pose.orientation.x = 0
                my_model.model_state.pose.orientation.y = 0
                my_model.model_state.pose.orientation.z = 0
                my_model.model_state.pose.orientation.w = 1
                my_model.model_state.twist.linear.x = 0.0
                my_model.model_state.twist.linear.y = 0.0
                my_model.model_state.twist.linear.z = 0.0
                my_model.model_state.twist.angular.x = 0.0
                my_model.model_state.twist.angular.y = 0.0
                my_model.model_state.twist.angular.z = 0.0
                my_model.model_state.reference_frame = "world"
                self.reset_proxy(my_model)
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_modelState service call failed")

    def resetWorld(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_world service call failed")

    def resetSim(self):
        rospy.wait_for_service("/gazebo/reset_simulation")
        try:
            self.reset_simulation()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_simulation service call failed")

    def pausePhysics(self):
        rospy.wait_for_service("/gazebo/pause_physics")  # 等待服务器连接
        try:
            self.pause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/pause_physics service call failed")

    def unpausePhysics(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/unpause_physics service call failed")

    def _model_states_cb(self, data):
        self.model_states = data
