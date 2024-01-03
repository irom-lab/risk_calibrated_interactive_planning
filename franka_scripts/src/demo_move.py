#!/usr/bin/env python3
"""

In order to grasp the yellow block, we would need to first find the yellow pixel from the RGB image, and then the corresponding point in the point cloud.

"""
import os
import sys
import numpy as np
import time
import scipy.interpolate
import rospy
import ropy as rp

from franka_irom_controllers.panda_commander import PandaCommander
from franka_irom_controllers.control_switcher import ControlSwitcher
from util.transform import quatMult, euler2quat, log_rot
from franka_irom.srv import GraspInfer2

from std_msgs.msg import Float64MultiArray, Bool, String
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
from sensor_msgs.msg import JointState
from demo import timeout, lm, process_mc_raw, temperature_scaling, mc_gen_prompt_template, mc_score_prompt_template

from util.transform import quatMult, euler2quat
# PYTHONPATH conflict... util folder from panda


class MoveEnv(object):
    def __init__(self):
        super(MoveEnv, self).__init__()

        # Initialize rospy node
        rospy.init_node('demo_move_env', anonymous=True)
        self.rate = rospy.Rate(1)  # Adjust the publication rate as needed

        # Set up panda moveit commander, pose control
        self.pc = PandaCommander(group_name='panda_arm')
        self.cs = ControlSwitcher({
            'moveit': 'position_joint_trajectory_controller',
            # 'velocity': 'joint_velocity_node_controller'
            })  # not initializing velocity controller
        self.cs.switch_controller('moveit')

        # Subscribe to robot state
        self.robot_state = None

        # Tracks data past for reset if repetitive
        # due to a needed high rate for camera node, camera node will likely be repetitve with data
        # Ensures that new data is being provided
        self.past_res = None




        # Initialize service from camera
        service_name = 'get_objs_pos'
        rospy.wait_for_service(service_name)
        self.objs_pos_srv = rospy.ServiceProxy(service_name, GraspInfer2)

        self.move_pub = rospy.Publisher('/move_flag', Bool, queue_size=1)
        
        


        # Errors
        self.ROBOT_ERROR_DETECTED = False
        self.BAD_UPDATE = False



    def __robot_state_callback(self, msg):
        self.robot_state = msg
        if any(self.robot_state.cartesian_collision):
            if not self.ROBOT_ERROR_DETECTED:
                rospy.logerr('Detected Cartesian Collision')
            self.ROBOT_ERROR_DETECTED = True
        for s in FrankaErrors.__slots__:
            if getattr(msg.current_errors, s):
                self.stop()
                if not self.ROBOT_ERROR_DETECTED:
                    rospy.logerr('Robot Error Detected')
                self.ROBOT_ERROR_DETECTED = True
                
        

    def open_gripper(self):
        self.pc.set_gripper(width=0.1)

    def close_gripper(self):
        self.pc.grasp(width=0.0, e_inner=0.1, e_outer=0.1, speed=0.05, 
            force=80)

    def run(self):
        
        # d = input("============ Maybe Press Enter to open gripper, press a to abort...")
        # if d == 'a':
        #     return
        # self.open_gripper()
        # time.sleep(0.1)

        # Loop 
        while not rospy.is_shutdown():

                # Extract block and bowl positions from res
            self.open_gripper()
            res = self.objs_pos_srv()
            block_pos = np.array([res.pos1.x, res.pos1.y, res.pos1.z])
            bowl_pos = np.array([res.pos2.x, res.pos2.y, res.pos2.z])
            obj_yaw = res.yaw
            if np.array_equal(block_pos,self.past_res): # check for redundant data due to high camera rate
                self.rate.sleep()
                continue
            if np.array_equal(block_pos,np.array([0,0,0])): # check for no contours detected
                # print('Contours not detected')
                # time.sleep(1)
                continue
                    
            d = input("============ Press Enter to move to initial pose, press b to abort...")
            if d == 'b':
                break
            initial_pos = np.array([0.3, 0, 0.5])
            initial_quat = quatMult(np.array([1.0, 0.0, 0.0, 0.0]), 
                                       euler2quat([np.pi/4,0,0]))
            initial_pose = list(np.concatenate((initial_pos, initial_quat)))
            self.pc.goto_pose(initial_pose, velocity=0.3)

            d = input("============ Press Enter to infer object pose from camera, press b to abort...")
            block_yaw = res.yaw
            print('Block position: ', block_pos)
            print('Block yaw: ', block_yaw)
            print('Bowl position: ', bowl_pos)
                # yaw isnt really that important for bowl (circular, symmetric)

            d = input("============ Press Enter to move to location above the grasp, press b to abort...")
            if d == 'b':
                break
            pick_target_pos_above = block_pos + np.array([0, 0, 0.3])
                # pick_target_pos_above = np.array([0.5, 0.1, 0.3])    # x,y,z
            target_quat = quatMult(np.array([1.0, 0.0, 0.0, 0.0]), 
                                       euler2quat([np.pi/4-obj_yaw,0,0]))
            pick_target_pose_above = list(np.concatenate((pick_target_pos_above, target_quat)))
            self.pc.goto_pose(pick_target_pose_above, velocity=0.25)

            d = input("============ Press Enter to move to pre grasp pose, press b to abort...")
            if d == 'b':
                break
                # target_pos = np.array([0.5, 0.1, 0.16])    # x,y,z
            pick_target_pos = block_pos + np.array([0, 0, 0.14])
                # the pos is for the end-effector, and not the tips of the gripper.
            pick_target_pose = list(np.concatenate((pick_target_pos, target_quat)))
            self.pc.goto_pose(pick_target_pose, velocity=0.15)

            d = input("============ Press Enter to grasp, press b to abort...")
            if d == 'b':
                break
            self.close_gripper()
            time.sleep(0.5)

            d = input("============ Press Enter to lift the block, press b to abort...")
            if d == 'b':
                break
            pick_target_pos_above += np.array([0, 0, .15]) #lift to a higher above postion to avoid collision with bowl
            pick_target_pose_above = list(np.concatenate((pick_target_pos_above, target_quat)))
            self.pc.goto_pose(pick_target_pose_above, velocity=0.20)

            d = input("============ Press Enter to put the block in the bowl, press b to abort...")
            if d == 'b':
                break
            put_target_pos = bowl_pos + np.array([0, 0, 0.2])
            put_target_pose = list(np.concatenate((put_target_pos, target_quat))) 
            self.pc.goto_pose(put_target_pose, velocity=0.20)

            d = input("============ Press Enter to open gripper, press b to abort...")
            if d == 'b':
                    break
            self.open_gripper()
            time.sleep(0.1)
            self.pc.goto_pose(initial_pose, velocity=0.3)
            self.move_pub.publish(Bool(data=True)) # confirms completed movement to interaction node allows for another prompt
            self.past_res = block_pos
            time.sleep(1)
                

           

        ################# Done #################
        
        


if __name__ == '__main__':
    move_env = MoveEnv()
    move_env.run()
