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
from franka_irom.srv import GraspInfer

from std_msgs.msg import Float64MultiArray
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
from sensor_msgs.msg import JointState

from util.transform import quatMult, euler2quat
# PYTHONPATH conflict... util folder from panda


class MoveEnv(object):
    def __init__(self):
        super(MoveEnv, self).__init__()

        # Initialize rospy node
        rospy.init_node('move_env', anonymous=True)
        self.ros_rate = 1000.0

        # Set up panda moveit commander, pose control
        self.pc = PandaCommander(group_name='panda_arm')
        self.cs = ControlSwitcher({
            'moveit': 'position_joint_trajectory_controller',
            # 'velocity': 'joint_velocity_node_controller'
            })  # not initializing velocity controller
        self.cs.switch_controller('moveit')

        # Subscribe to robot state
        self.robot_state = None

        # Initialize service from camera
        service_name = 'get_obj_pos'
        rospy.wait_for_service(service_name)
        self.obj_pos_srv = rospy.ServiceProxy(service_name, GraspInfer)

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

        d = input("============ Press Enter to open gripper, press a to abort...")
        if d == 'a':
            return
        self.open_gripper()
        time.sleep(0.1)

        # Loop scoops
        while not rospy.is_shutdown():

            d = input("============ Press Enter to move to initial pose, press a to abort...")
            if d == 'a':
                break
            initial_pos = np.array([0.5, 0, 0.5])
            initial_quat = quatMult(np.array([1.0, 0.0, 0.0, 0.0]), 
                                   euler2quat([np.pi/4,0,0]))
            initial_pose = list(np.concatenate((initial_pos, initial_quat)))
            self.pc.goto_pose(initial_pose, velocity=0.3)

            d = input("============ Type an x position and press Enter, press a to abort...")
            if d == 'a':
                break
            pos1_x = float(d)
            while(pos1_x < 0.2):
                d = input("============ Type an x position of at least 0.2 and press Enter, press a to abort...")
                if d == 'a':
                    break
                pos1_x = float(d)


            d = input("============ Type a y position and press Enter, press a to abort...")
            pos1_y = float(d)
            if d == 'a':
                break

            d = input("============ Type a z position and press Enter, press a to abort")
            pos1_z = float(d)
            if d == 'a':
                break

            d = input("============ Type a yaw and press Enter, press a to abort")
            pos1_yaw = float(d)
            if d == 'a':
                break

            initial_pos = np.array([pos1_x, pos1_y, pos1_z])
            target_quat = quatMult(np.array([1.0, 0.0, 0.0, 0.0]), 
                                   euler2quat([np.pi/4-pos1_yaw,0,0]))
            initial_pose = list(np.concatenate((initial_pos, initial_quat)))
            self.pc.goto_pose(initial_pose, velocity=0.3)

        ################# Done #################
        d = input("========= Press enter to finish demo")


if __name__ == '__main__':
    move_env = MoveEnv()
    move_env.run()
