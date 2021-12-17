import copy
import rospy
from std_msgs.msg import Float32MultiArray, Bool
from gazebo_msgs.msg import  ModelStates
from geometry_msgs.msg import Vector3
import time
import numpy as np

from PIDClass import PID
from jacobi_linearprog import jacobi

'''
TODO: 以500hz计算绳子拉力
'''

class cal_force():
    def __init__(self) -> None:
        rospy.init_node('force_cal')

        # Subscriber
        rospy.Subscriber('/cmd_running', Bool, self.sub_run)
        rospy.Subscriber('/uavs_posi', Float32MultiArray, self.sub_uavs_posi)
        rospy.Subscriber('/cmd_ball_posi', Vector3, self.sub_ball_posi)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.sub_state)

        # Publisher
        self.force_pub = rospy.Publisher('/cmd_force', Float32MultiArray, queue_size=1)

        # define variables
        self.force = Float32MultiArray()

        # initial variables
        self.point_end_effector = np.array([[0, 0, 0] for _ in range(7)])
        self.payload_gravity = 0.98
        self.force.data = [0 for i in range(19)]
        self.running = False
        self.cmd_ball_posi = None
        self.payload_posi = None
        self.logger_posi = None
        self.payload_v = None
        self.uavs_posi = None

        self.count = 0

        rospy.loginfo('creating a node to calculate force by 200hz')

    def pub(self):
        force_controller = apmPID()
        rate = rospy.Rate(200)
        start = time.time()
        while not rospy.is_shutdown():
            if not self.running:
                rate.sleep()
                continue

            # print('payload_posi: ', self.payload_posi[2])

            print('time:{} '.format(time.time() - start))

            start = time.time()
            # TODO: calculate the force by pid
            force_controller.set_target(self.cmd_ball_posi)
            cable_tensions = self.compute_force(force_controller)
            
            self.force.data = [
                cable_tensions[4],  cable_tensions[5], cable_tensions[6],
                cable_tensions[0],  cable_tensions[1], cable_tensions[2], cable_tensions[3],
                self.uavs_posi[0], self.uavs_posi[1], self.uavs_posi[2],
                self.uavs_posi[3], self.uavs_posi[4], self.uavs_posi[5],
                self.uavs_posi[6], self.uavs_posi[7], self.uavs_posi[8],
                self.uavs_posi[9], self.uavs_posi[10], self.uavs_posi[11],
            ]
            self.force_pub.publish(self.force)
            print('z: ',self.payload_posi[2])
            rate.sleep()


    def compute_force(self, pid_controller):
        target_force_wrench = pid_controller.cal_actions(self.payload_posi)
        
        # print('target_force_wrench: ', target_force_wrench)
        target_force_wrench = target_force_wrench + np.array([0., 0., self.payload_gravity])
        points_base = np.concatenate((np.reshape(self.uavs_posi, (4,3)), self.logger_posi), axis=0)
        point_end_effector = self.point_end_effector
        pose_0 = np.array([0.,0.,0.,1.0])
        rotation_center = self.payload_posi
        # cable_one_side = np.array([rotation_center for i in range(7)])
        Jac, cable_tensions, force_wrench = jacobi(points_base, point_end_effector, pose_0, rotation_center, target_force_wrench)
        # print('cable_tensions ', cable_tensions)
        return cable_tensions

    def sub_ball_posi(self, data):
        self.cmd_ball_posi = np.array([data.x, data.y, data.z])

    def sub_uavs_posi(self, data):
        self.uavs_posi = np.array(data.data)

    def sub_state(self, data):
        payloads_index = data.name.index('payload')
        logger0_index = data.name.index('omni_car_0')
        logger1_index = data.name.index('omni_car_1')
        logger2_index = data.name.index('omni_car_2')

        self.payload_posi, self.payload_v = self.get_posi_v(data,payloads_index)
        logger0_posi, _ = self.get_posi_v(data,logger0_index)
        logger1_posi, _ = self.get_posi_v(data,logger1_index)
        logger2_posi, _ = self.get_posi_v(data,logger2_index)
        self.logger_posi = np.array([logger0_posi, logger1_posi, logger2_posi])

        
        # if self.payload_posi[2] < 0.1:
        #     self.count += 1


    def get_posi_v(self, data, index):
        pose = data.pose[index]
        twist = data.twist[index]
        posi = np.array(
            [pose.position.x, pose.position.y, pose.position.z])
        v = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
        return posi, v

    def sub_run(self, data):
        self.running = data.data


class apmPID:
    def __init__(self, target=np.array([0, 0, 0])):
        x_p, x_i, x_d = 0.08, 0, 0.05
        y_p, y_i, y_d = 0.08, 0, 0.05
        z_p, z_i, z_d = 0.1, 0.0, 0.

        # 倍数
        x, y, z = 2.5, 2.5, 1

        self.control_x = PID(np.asarray([x_p, x_i, x_d]) * x, target[0], upper=100,
                             lower=-100)  # control position x
        self.control_y = PID(np.asarray([y_p, y_i, y_d]) * y, target[1], upper=100,
                             lower=-100)  # control position y
        self.control_z = PID(np.asarray([z_p, z_i, z_d]) * z, target[2], upper=100,
                             lower=-100)  # control position z

    def cal_actions(self, state):
        '''
        :param state: 目标位姿
        :return: 力+力矩
        '''
        u1 = self.control_x.cal_output(state[0])
        u2 = self.control_y.cal_output(state[1])
        u3 = self.control_z.cal_output(state[2])
        U = np.array([u1, u2, u3])
        # print('current state: ',state)
        # print('pid output: ',list)
        return U

    def set_target(self, target):
        self.control_x.set_target(target=target[0])
        self.control_y.set_target(target=target[1])
        self.control_z.set_target(target=target[2])


if __name__ == '__main__':
    cal = cal_force()
    cal.pub()
