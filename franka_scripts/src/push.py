#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
import scipy.interpolate
import rospy
import ropy as rp

from franka_irom_controllers.panda_commander import PandaCommander
from franka_irom_controllers.control_switcher import ControlSwitcher
from util.transform import quatMult, euler2quat, log_rot, quat2list
from util.trajectory import QuinticTimeScaling, LinearTimeScaling

from std_msgs.msg import Float64MultiArray
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from franka_irom.srv import Pos, TriggerRecord

from pydrake.all import RigidTransform, RotationMatrix, RollPitchYaw


class PushEnv(object):
    def __init__(self):
        super(PushEnv, self).__init__()

        # Initialize rospy node
        rospy.init_node('push_env', anonymous=True)

        # Set up panda moveit commander, pose control
        self.pc = PandaCommander(group_name='panda_arm')

        # Set up ropy and joint velocity controller
        self.panda = rp.Panda()
        self.curr_velocity_publish_rate = 500.0  # for libfranka
        self.curr_velo_pub = rospy.Publisher(
                            '/joint_velocity_node_controller/joint_velocity', 
                            Float64MultiArray, 
                            queue_size=1)
        self.curr_velo = Float64MultiArray()

        # Set up switch between moveit and velocity, start with moveit
        self.cs = ControlSwitcher({
            'moveit': 'position_joint_trajectory_controller',
            'velocity': 'joint_velocity_node_controller'
            })
        self.cs.switch_controller('moveit')

        # Subscribe to robot state
        self.robot_state = None
        rospy.Subscriber('/joint_states', 
                        JointState, 
                        self.__joint_state_callback, 
                        queue_size=10)

        # Initialize service from cameraEnv
        service_name = '/push_camera/get_bottle_pos'
        rospy.wait_for_service(service_name)
        self.bottle_pos_srv = rospy.ServiceProxy(service_name, Pos)

        # Initialize service for rs and k4a recording
        self.flag_recording_rs = 'True' in sys.argv[2]
        self.flag_recording_k4a = 'True' in sys.argv[3]
        print(f'Recording rs? {self.flag_recording_rs}. Recording k4a? {self.flag_recording_k4a}')
        if self.flag_recording_rs:
            rs_service_name = '/rs_record/trigger_record'
            rospy.wait_for_service(rs_service_name)
            self.rs_trigger_record_srv = rospy.ServiceProxy(rs_service_name, TriggerRecord)
        if self.flag_recording_k4a:
            k4a_service_name = '/k4a_record/trigger_record'
            rospy.wait_for_service(k4a_service_name)
            self.k4a_trigger_record_srv = rospy.ServiceProxy(k4a_service_name, TriggerRecord)

        # Errors
        self.ROBOT_ERROR_DETECTED = False
        self.BAD_UPDATE = False

        # A few useful joint angles/transforms
        self.q_init = [-0.084, -0.415, 0.0818, -2.584, 0.041, 2.169, 0.754] # [0.40, 0.0, 0.40]
        self.q_push_init_above = [9.18724181e-05, 1.08386303e-01, -8.31188463e-05, -2.78074937e+00, 3.65240674e-05, 2.88912814e+00, 7.85e-01]   # 0.18
        self.R_E_fixed = RotationMatrix(RollPitchYaw(0, -np.pi, np.pi/2)).multiply(RotationMatrix(RollPitchYaw(0, 0, -np.pi/4))) # Fixed transform from world to EE pointing down


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


    def __joint_state_callback(self, msg):
        self.joint_state = msg


    def stop(self):
        msg = Float64MultiArray()
        msg.data = [0., 0., 0., 0., 0., 0., 0.]
        self.curr_velo_pub.publish(msg)


    def get_bottle_2d(self):
        res = self.bottle_pos_srv()
        pos = np.array([res.pos.x, res.pos.y])
        return pos


    def run(self):

        # Parameterize pushing trajectory
        t1 = 0.2
        t2 = 0.1
        t3 = 0.5
        t_total = t1 + t2 + t3
        yaw = 0
        vel_x = 0.6
        vel_pitch = -0.8
        vel_z = 0.1
        vel_init = np.zeros((6, 1))
        bottle_initial_x = 0.56
        offset = 0.15   # was 0.07
        max_dist_bottle_goal = 0.2
        num_trial = int(sys.argv[1])
        video_folder_prefix = sys.argv[4]
        cnt_trial = 0
        trial_final_T = np.empty((0,2))
        trial_reward = []

        # Build folder is not exist
        if not os.path.isdir(video_folder_prefix):
            os.makedirs(video_folder_prefix)

        # Initial pose
        d = input("============ Press Enter to move to start pose, press a to skip...")
        if d != 'a':
            self.pc.goto_joints(self.q_init, velocity=0.5)

        # Intermediate pose
        d = input("============ Press Enter to move to intermediate pose, press a to abort...")
        if d == 'a':
            return
        self.pc.goto_joints(self.q_push_init_above, velocity=0.5)

        # Loop scoops
        while not rospy.is_shutdown():

            # Input forward velocity and yaw
            while 1:
                try:
                    vel_x = float(input("============ Enter x_dot..."))
                except:
                    continue
                else:
                    break
            while 1:
                try:
                    yaw = float(input("============ Enter yaw..."))
                except:
                    continue
                else:
                    break

            # Figure out EE pose based on yaw input
            R_E = RotationMatrix(RollPitchYaw(0, 0, yaw))
            T_E = [bottle_initial_x - offset*np.cos(yaw), 
                    -offset*np.sin(yaw), 
                    0.135]
            R_E = R_E.multiply(self.R_E_fixed)
            push_pose = T_E + quat2list(R_E.ToQuaternion())

            # Figure out top velocity based on velocity and yaw input
            vel_fast = np.array([vel_x*np.cos(yaw), 
                                vel_x*np.sin(yaw), 
                                vel_z,
                                -vel_pitch*np.sin(yaw), 
                                vel_pitch*np.cos(yaw), 
                                0, 
                                ]).reshape((6, 1))

            # Move closer
            d = input("============ Press Enter to move to final pose before pushing, press a to abort...")
            if d == 'a':
                break
            # self.pc.goto_joints(self.q_push_init, velocity=0.1)
            self.pc.goto_pose(push_pose, velocity=0.1)

            # Switch
            d = input("============ Press Enter to push, press a to abort...")
            if d == 'a':
                break

            # Start video
            if self.flag_recording_rs:
                res = self.rs_trigger_record_srv(os.path.join(video_folder_prefix, str(cnt_trial)+'_rs.bag'))
            if self.flag_recording_k4a:
                res = self.k4a_trigger_record_srv(os.path.join(video_folder_prefix, str(cnt_trial)+'_k4a.bag'))

            self.cs.switch_controller('velocity')
            time.sleep(0.1)
            rate = rospy.Rate(self.curr_velocity_publish_rate)
            for _ in range(10):
                rate.sleep()
            self.panda.q = np.array(self.joint_state.position[:7])
            for _ in range(100):
                d = self.panda.jacob0()

            # Time
            t_init = rospy.get_rostime().to_sec()
            t_cur = 0
            t_prev = -0.002 # arbitrary

            # Segment
            flag_t1 = False
            flag_t2 = False
            flag_t3 = False
            while t_cur <= t_total:

                # Get current joint angles and current cartesian velocity from jacobian
                self.panda.q = np.array(self.joint_state.position[:7])
                dq = np.array(self.joint_state.velocity[:7])
                jac = self.panda.jacob0()
                v = jac.dot(dq)

                # Find target vel
                t_cur = rospy.get_rostime().to_sec() - t_init
                if t_cur > t_total:
                    break
                elif t_cur < t1:
                    if not flag_t1:
                        traj = scipy.interpolate.interp1d([t_cur, t1], np.hstack((v[:, None], vel_fast)))
                        flag_t1 = True
                elif t_cur < t1+t2:
                    if not flag_t2:
                        traj = scipy.interpolate.interp1d([t_cur, t1+t2], np.hstack((v[:, None], vel_fast)))
                        flag_t2 = True
                else:
                    if not flag_t3:
                        traj = scipy.interpolate.interp1d([t_cur, t_total], np.hstack((v[:, None], vel_init)))
                        flag_t3 = True
                v_d = traj(t_cur)

                # Damped differential inverse kinematics
                damping = np.eye((6))*0.002
                pinv = jac.T.dot(np.linalg.inv(jac.dot(jac.T) + damping))
                dq_d = pinv.dot(v_d)

                # Send joint velocity cmd
                msg = Float64MultiArray()
                msg.data = dq_d
                self.curr_velo_pub.publish(msg)
                rate.sleep()

            # Send zero velocities for a bit longer in case the arm has not stoped completely yet
            ctr = 0
            while ctr < 100:
                self.stop()
                ctr += 1
                rate.sleep()

            # Switch back to pose control
            self.cs.switch_controller('moveit')

            # Stop video
            if self.flag_recording_rs:
                time.sleep(0.5)
                res = self.rs_trigger_record_srv('stop')
            if self.flag_recording_k4a:
                time.sleep(0.5)
                res = self.k4a_trigger_record_srv('stop')

            # Move arm away
            d = input("============ Press Enter to move to intermediate pose, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(self.q_push_init_above, velocity=0.3)

            # Get bottle pos
            while 1:
                d = input("============ Press Enter to grab depth...")
                bottle_T_final = self.get_bottle_2d()
                print('Bottle T: ', bottle_T_final)
                d = input("============ Press Enter to accept, press a to re-grab...")
                if not d == 'a':
                    break

            # Get reward
            while 1:
                try:
                    goal_x = float(input("============ Enter goal_x..."))
                except:
                    continue
                else:
                    break
            while 1:
                try:
                    goal_y = float(input("============ Enter goal_y..."))
                except:
                    continue
                else:
                    break
            goal = np.array([goal_x, goal_y])
            dist_bottle_goal = np.linalg.norm(np.array(bottle_T_final[:2])-goal)
            dist_ratio_bottle_goal = dist_bottle_goal / max_dist_bottle_goal
            reward = max(0, 1-dist_ratio_bottle_goal)
            print("=== Trial Summary ===")
            print("=== Action: ", (vel_x, yaw))
            print("=== Goal: ", (goal_x, goal_y))
            print("=== Result: ", (bottle_T_final[0], bottle_T_final[1]))
            print("=== Reward: ", reward)
            print("=====================")

            # Finalize
            d = input("============ Press Enter to accept trial, press a to reject...")
            if not d == 'a':
                cnt_trial += 1

                trial_final_T = np.vstack((trial_final_T, bottle_T_final))
                trial_reward += [reward]

                if cnt_trial == num_trial:
                    print('All final T:')
                    print(np.array2string(trial_final_T, separator=', '))
                    print('Average reward: ')
                    print(np.mean(trial_reward))
                    cnt_trial = 0
                    trial_final_T = np.empty((0,2)) 
                    trial_reward = []

        # Finish
        d = input("========= Press enter to finish demo")


if __name__ == '__main__':
    push_env = PushEnv()
    push_env.run()
