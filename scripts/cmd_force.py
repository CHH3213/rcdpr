import rospy
from std_msgs.msg import Float32MultiArray, Bool
import time

class cmd_force():
    def __init__(self) -> None:
        rospy.init_node('force_pub')

        # Subscriber
        rospy.Subscriber('/cmd_running', Bool, self.sub_run)
        rospy.Subscriber('/cmd_force', Float32MultiArray,self.sub_force)

        # Publisher
        self.force_pub = rospy.Publisher('/rcdpr_force', Float32MultiArray, queue_size=1)

        # define variables
        self.force = Float32MultiArray()

        # initial variables
        self.running = False
        self.force.data = [0 for i in range(19)]

        rospy.loginfo('creating a node to publish force by 1000hz')

    def pub(self):
        rate = rospy.Rate(1100)
        while not rospy.is_shutdown():
            if not self.running:
                rate.sleep()
                self.force.data = [0 for i in range(19)]
                continue
            start = time.time()
            # print('force: {}\n'.format(self.force.data))
            self.force_pub.publish(self.force)
            rate.sleep()
            end = time.time()
            print('time: ', end - start)

    def sub_force(self,data):
        self.force = data

    def sub_run(self, data):
        self.running = data.data

if __name__ == '__main__':
    cmd = cmd_force()
    cmd.pub()
