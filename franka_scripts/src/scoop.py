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
from util.transform import quatMult, euler2quat, log_rot
from util.trajectory import QuinticTimeScaling, LinearTimeScaling
from franka_irom.srv import Pos, TriggerRecord

from std_msgs.msg import Float64MultiArray
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
from sensor_msgs.msg import JointState

from pydrake.all import RigidTransform, RotationMatrix, RollPitchYaw
from env.scoop_exp_env import ScoopExpEnv   # yikes... gonna fix the paths later
import pickle
# PYTHONPATH conflict... util folder from panda


class ScoopEnv(object):
    def __init__(self):
        super(ScoopEnv, self).__init__()

        # Initialize rospy node
        rospy.init_node('scoop_env', anonymous=True)
        self.ros_rate = 1000.0

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
                        queue_size=1)

        # Initialize service from cameraEnv
        # service_name = '/push_camera/get_bottle_pos'
        # rospy.wait_for_service(service_name)
        # self.bottle_pos_srv = rospy.ServiceProxy(service_name, Pos)

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

        # Constants
        self.q_init = [0, 0.0769, 0, -2.726, 0, 2.803, 0.786]
        self.q_release = [-0.017, 0.630, -0.020, -2.280, 0.015, 3.303, 0.805]

        # Initialize panda environment in Drake
        dataset = '/home/allen/panda/data/veggie_task/v3_cylinder_single/1000.pkl'
        print("= Loading tasks from", dataset)
        with open(dataset, 'rb') as f:  # use any task
            self.sim_task = pickle.load(f)[0]
        self.sim_env = ScoopExpEnv(dt=0.005,
                                    render=False,
                                    visualize_contact=False,
                                    camera_param=None,
                                    hand_type='panda',
                                    diff_ik_filter_hz=200,
                                    contact_solver='sap',
                                    panda_joint_damping=1.0,
                                    table_type='normal',
                                    flag_disable_rate_limiter=True,
                                    num_link=1,
                                    # veggie_x=0.68,
                                    traj_type='relative_to_tip'
                                    )


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


    # def get_bottle_2d(self):
    #     res = self.bottle_pos_srv()
    #     pos = np.array([res.pos.x, res.pos.y])
    #     return pos
    

    def open_gripper(self):
        self.pc.set_gripper(width=0.1)


    def close_gripper(self):
        self.pc.grasp(width=0.0, e_inner=0.1, e_outer=0.1, speed=0.05, 
            force=80)


    def run(self):
        num_trial = int(sys.argv[1])
        video_folder_prefix = sys.argv[4]
        cnt_trial = 0
        trial_reward = []

        # Action space
        # s_pitch_init = 5*np.pi/180
        # pd_1 = -0.007
        # s_x_veggie_tip = -0.01

        # deg=6, s_x=-0.01, radius=0.005
        # q_pre_grasp =  [-0.229, 0.376,-0.135,-2.093,-0.086, 2.154, 0.467]
        # q_pre_grasp_top =  [-0.203, 0.29 ,-0.133,-1.809,-0.077, 1.783, 0.463]
        # q_pre_grasp_back =  [-0.214, 0.244,-0.156,-2.117,-0.097, 2.047, 0.458]
        # q_pre_scoop_up =  [-0.001, 0.325, 0.001,-2.418, 0.001, 2.75 , 0.784]
        # q_pre_scoop =  [ 0.008, 0.38 ,-0.007,-2.408, 0.009, 2.795, 0.779]
        # deg=7, -0.007, -0.01, radius=0.005
        # q_pre_grasp =  [-0.229, 0.376,-0.135,-2.093,-0.086, 2.154, 0.467]
        # q_pre_grasp_top =  [-0.203, 0.29 ,-0.133,-1.809,-0.077, 1.783, 0.463]
        # q_pre_grasp_back =  [-0.214, 0.244,-0.156,-2.117,-0.097, 2.047, 0.458]
        # q_pre_scoop_up =  [-0.002, 0.315, 0.002,-2.418,-0.   , 2.74 , 0.785]
        # q_pre_scoop =  [ 0.006, 0.37 ,-0.005,-2.408, 0.007, 2.785, 0.78]

        d = input("============ Press Enter to open gripper, press a to abort...")
        if d == 'a':
            return
        self.open_gripper()
        time.sleep(0.1)

        # Loop scoops
        while not rospy.is_shutdown():

            d = input("============ Press Enter to move to start pose, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(self.q_init, velocity=0.4)

            # Input action
            while 1:
                try:
                    s_pitch_init = float(input("============ Enter s_pitch_init..."))
                except:
                    continue
                else:
                    break
            while 1:
                try:
                    pd_1 = float(input("============ Enter pd_1..."))
                except:
                    continue
                else:
                    break
            while 1:
                try:
                    s_x_veggie_tip = float(input("============ Enter s_x_veggie_tip..."))
                except:
                    continue
                else:
                    break

            # Calculate trajectory from sim env
            veggie_radius = 0.005
            self.sim_env.reset(self.sim_task)
            q_all = self.sim_env.step([s_pitch_init, pd_1, s_x_veggie_tip, veggie_radius])
            q_pre_grasp = q_all['q_pre_grasp']
            q_pre_grasp_top = q_all['q_pre_grasp_top']
            q_pre_grasp_back = q_all['q_pre_grasp_back']
            q_pre_scoop_up = q_all['q_pre_scoop_up']
            q_pre_scoop = q_all['q_pre_scoop']
            q_pre_dump = q_all['q_pre_dump']
            q_dump = q_all['q_dump']
            q_post_dump = q_all['q_post_dump']

            # Time for spline 
            tn = [0, 0.25, 0.1, 0.25, 0.1, 0.3]
            tn = np.cumsum(tn)
            t_total = tn[-1]

            # Spline for x direction
            xd = np.zeros((6))
            xd[1] = 0.65
            xd[2] = 0.75
            xd[3] = 0.2
            s_xd = scipy.interpolate.CubicSpline(tn, xd, bc_type='clamped')
            # s_xd = scipy.interpolate.interp1d(tn, xd)

            # Spline for z direction
            # zd = [0, -0.01, 0.02, 0.02, 0, 0]
            zd = [0, 0, 0.01, 0.02, 0, 0]   # was -0.01 for 2nd
            # zd = [0, -0.01, -0.01, 0.01, 0, 0]
            s_zd = scipy.interpolate.CubicSpline(tn, zd, bc_type='clamped')
            # s_zd = scipy.interpolate.interp1d(np.append(tn, 1.1), zd+[0]) # otherwise extrapolation error

            # Spline for pitch direction
            pitchd = np.zeros((6))
            pitchd[1] = pd_1
            pitchd[2:4] = -0.6
            s_pitchd = scipy.interpolate.CubicSpline(tn, pitchd, bc_type='clamped')
            # s_pitchd = scipy.interpolate.interp1d(tn, pitchd)

            ################## Reach the holder and grasp ##################

            d = input("============ Press Enter to move to pre grasp <<top>> pose, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp_top, velocity=0.5)

            d = input("============ Press Enter to move to pre grasp pose, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp, velocity=0.2)

            d = input("============ Press Enter to grasp, press a to abort...")
            if d == 'a':
                break
            self.close_gripper()
            time.sleep(0.5)

            d = input("============ Press Enter to move to pre grasp <<back>> pose, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp_back, velocity=0.2)

            ################## Move spatula to table ##################

            # d = input("============ Press Enter to move to initial pose, press a to abort...")
            # init_joint_angles = [0, 0.151, 0, -2.544, 0, 2.695, 0.786]	# 0, 5, -0.05
            # self.pc.goto_joints(init_joint_angles, velocity=0.3)

            d = input("============ Press Enter to move to pre scoop <<up>> pose, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_scoop_up, velocity=0.5)

            d = input("============ Press Enter to move to pre scoop pose, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_scoop, velocity=0.02)

            ################## Scoop ##################

            d = input("============ Press Enter to scoop, press a to abort...")
            if d == 'a':
                break

            # Start video
            if self.flag_recording_rs:
                res = self.rs_trigger_record_srv(os.path.join(video_folder_prefix, str(cnt_trial)+'_rs.bag'))
            if self.flag_recording_k4a:
                res = self.k4a_trigger_record_srv(os.path.join(video_folder_prefix, str(cnt_trial)+'_k4a.bag'))

            # Switch controller
            self.cs.switch_controller('velocity')
            time.sleep(0.1)
            rate = rospy.Rate(self.ros_rate)
            for _ in range(10):
                rate.sleep()
            self.panda.q = np.array(self.joint_state.position[:7])
            for _ in range(100):
                d = self.panda.jacob0()

            # Record
            # dq_d_all = []
            # dq_all = []
            # ddq_all = []
            # dddq_all = []
            # vel_d_all = []
            # vel_all = []
            # dq_prev = np.zeros((7))
            # ddq_prev = np.zeros((7))
            # t_prev = -0.002 # arbitrary
            t_init = rospy.get_rostime().to_sec()
            t_cur = 0

            # Segment
            while t_cur <= t_total:

                # Get current joint angles and current cartesian velocity from jacobian
                self.panda.q = np.array(self.joint_state.position[:7])
                dq = np.array(self.joint_state.velocity[:7])
                jac = self.panda.jacob0()
                v = jac.dot(dq)

                # Find target vel
                t_cur = rospy.get_rostime().to_sec() - t_init
                # rospy.loginfo(t_cur)
                v_d = [s_xd(t_cur), 0, s_zd(t_cur), 0, s_pitchd(t_cur), 0]

                # Damped differential inverse kinematics
                damping = np.eye((6))*0.002
                pinv = jac.T.dot(np.linalg.inv(jac.dot(jac.T) + damping))
                dq_d = pinv.dot(v_d)

                # Record
                # dq_all += [dq]
                # dq_d_all += [dq_d]
                # ddq = (dq - dq_prev)/(t_cur-t_prev)
                # ddq_all += [ddq]
                # dddq = (ddq - ddq_prev)/(t_cur-t_prev)
                # dddq_all += [dddq]
                # vel_d_all += [v_d]
                # vel_all += [v]
                # dq_prev = dq
                # ddq_prev = ddq
                # t_prev = t_cur

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

            # Plot
            # dq_d_all = np.vstack(dq_d_all)
            # dq_all = np.vstack(dq_all)
            # ddq_all = np.vstack(ddq_all)
            # dddq_all = np.vstack(dddq_all)
            # vel_d_all = np.vstack(vel_d_all)
            # vel_all = np.vstack(vel_all)
            # import matplotlib.pyplot as plt
            # f, axarr = plt.subplots(1,7)
            # for joint_ind in range(7):
            #     axarr[joint_ind].plot(dq_d_all[:,joint_ind], label='desired')
            #     axarr[joint_ind].plot(dq_all[:,joint_ind], label='actual')
            #     axarr[joint_ind].legend()
            # plt.show()

            # f, axarr = plt.subplots(6,1)
            # for ind in range(6):
            #     axarr[ind].plot(vel_d_all[:,ind], label='desired')
            #     axarr[ind].plot(vel_all[:,ind], label='actual')
            #     axarr[ind].legend() 
            # plt.show()

            d = input("============ Press Enter to dump, press a to abort...")
            if d == 'a':
                break

            # Dump
            self.pc.goto_joints(q_pre_dump, velocity=0.5)
            self.pc.goto_joints(q_dump, velocity=0.5)
            time.sleep(0.5)
            self.pc.goto_joints(q_post_dump, velocity=0.5)

            # Stop video
            if self.flag_recording_rs:
                time.sleep(0.5)
                res = self.rs_trigger_record_srv('stop')
            if self.flag_recording_k4a:
                time.sleep(0.5)
                res = self.k4a_trigger_record_srv('stop')

            d = input("============ Press Enter to move down and release spatula, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(self.q_release, velocity=0.3)

            # Open gripper
            self.open_gripper()
            time.sleep(0.1)

            while 1:
                try:
                    reward = float(input("============ Enter reward..."))
                except:
                    continue
                else:
                    break

            # Finalize
            d = input("============ Press Enter to accept trial, press a to reject...")
            if not d == 'a':
                cnt_trial += 1

                trial_reward += [reward]

                if cnt_trial == num_trial:
                    print('Average reward: ')
                    print(np.mean(trial_reward))
                    cnt_trial = 0
                    trial_reward = []

            # Back to home
            d = input("============ Press Enter to home, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(self.q_init, velocity=0.5)

        ################# Done #################

        # Finish
        d = input("========= Press enter to finish demo")


if __name__ == '__main__':
    scoop_env = ScoopEnv()
    scoop_env.run()
