#!/usr/bin/env python3
from std_msgs.msg import String, Bool
from franka_irom.srv import SentText, SentTextResponse
import rospy
import time

class NodeTwoEnv(object):
    def __init__(self):
      super(NodeTwoEnv,self).__init__()
      self.rev_text = ""
      self.rate = rospy.Rate(10)  # slow publication rate allows for node 3 to request and recieve srv before rev_text is cleared
        
        # listens async in background for node1 data which is used as a flag
      rospy.Subscriber('/node_1_data', String, self.__text_callback) 
      

        # Setup text reversal service
      text_srv = rospy.Service('get_rev_text',SentText, self.get_rev_text)

    
      

    def __text_callback(self,msg):
            self.text = msg.data # recieved string
            self.rev_text = self.text[::-1]
            
            

    # string reversal service
    def get_rev_text(self,req):
        sendingText = SentTextResponse() 
        sendingText.mainString = self.rev_text
        return sendingText


        

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

                
                        

if __name__ == '__main__':
    rospy.init_node('node_two_env',anonymous=True)
    nodeTwo = NodeTwoEnv()
    nodeTwo.run()
