#!/usr/bin/env python3
from std_msgs.msg import String, Bool
import rospy

class NodeOneEnv(object):
    def __init__(self):
        super(NodeOneEnv, self).__init__()
        self.prompted = False
        self.pub1 = rospy.Publisher('/node_1_data', String, queue_size=10)
        self.rate = rospy.Rate(1)  # Adjust the publication rate as needed
        rospy.Subscriber('/node_3_flag', Bool, self.__prompt_callback) # listens async in background for flags
        

    def __prompt_callback(self,msg):
            self.prompted = msg.data # strictly True or False
        

    def recieve_input(self):
            d = input("Type something: ")
            user_output = String(data=d)
            self.pub1.publish(user_output)


    def run(self):
        while not rospy.is_shutdown():
                if self.prompted:
                    self.recieve_input()
                    self.prompted = False
                else:
                    # async background check for call_backs
                    self.rate.sleep()
                                
                        

if __name__ == '__main__':
    rospy.init_node('node_one_env',anonymous=True)
    nodeOne = NodeOneEnv()
    nodeOne.recieve_input()
    nodeOne.run()
