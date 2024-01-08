#!/usr/bin/env python3
from std_msgs.msg import String, Bool
from franka_irom.srv import SentText
import rospy
import time


class NodeThreeEnv(object):
    def __init__(self):
      super(NodeThreeEnv,self).__init__()
      self.flag = False
      self.pub3 = rospy.Publisher('/node_3_flag', Bool, queue_size=1) # sends flag to node 1
      self.rate = rospy.Rate(1)  # Adjust the publication rate as needed

      self.rev_txt_past = ""
         
        
        # Initialize textReversal srv
      service_name = 'get_rev_text'
      rospy.wait_for_service(service_name)
      self.rev_txt_srv = rospy.ServiceProxy(service_name, SentText)
        
        
       

    def __flag_callback(self,msg):
            self.flag = msg.data # strictly True or False
    

  
    def run(self):
        while not rospy.is_shutdown():  
            sendingText = self.rev_txt_srv()
            total_text = sendingText.mainString
            if (total_text == self.rev_txt_past):
                  self.rate.sleep()
                  continue         

            else:
                # async background check for call_backs
                print("success")
                print(total_text+ " " + str(len(total_text))) # Print reversed text with string length
                self.rev_txt_past = total_text
                self.pub3.publish(Bool(data=True))  # truth flag to node 1 verifies completion and allows for next input
            
    
                
                        

if __name__ == '__main__':
    rospy.init_node('node_three_env',anonymous=True)
    nodeThree = NodeThreeEnv()
    nodeThree.run()
