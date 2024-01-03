#!/usr/bin/env python3
import os
import sys
sys.dont_write_bytecode = True

import numpy as np
import time
import scipy.interpolate

import rospy
import ropy as rp

from franka_irom_controllers.panda_commander import PandaCommander
from franka_irom_controllers.control_switcher import ControlSwitcher
from util.transform import quatMult, euler2quat, log_rot
from util.trajectory import QuinticTimeScaling, LinearTimeScaling

from std_msgs.msg import Float64MultiArray
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
from sensor_msgs.msg import JointState


class ScoopEnv(object):
    def __init__(self):
        super(ScoopEnv, self).__init__()

        # Initialize rospy node
        rospy.init_node('scoop_env', anonymous=True)

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


    def __joint_state_callback(self, msg):
        self.joint_state = msg


    def stop(self):
        msg = Float64MultiArray()
        # msg = Twist()
        msg.data = [0., 0., 0., 0., 0., 0., 0.]
        self.curr_velo_pub.publish(msg)


    def go(self):

        d = input("============ Press Enter to open gripper, press a to abort...")
        if d == 'a':
            return
        self.pc.set_gripper(width=0.1)
        time.sleep(0.1)

        # Loop scoops
        while not rospy.is_shutdown():

            d = input("============ Press Enter to move to start pose, press a to abort...")
            if d == 'a':
                return
            start_joint_angles = [ 6.16153856e-05,  7.68869047e-02 ,-5.86934186e-05, -2.72579434e+00, 1.34655064e-05 , 2.80267525e+00  ,7.85786519e-01]
            self.pc.goto_joints(start_joint_angles, velocity=0.2)
            # start_pos = [0.40, 0.0, 0.20]
            # # start_quat = list(quatMult(np.array([1.0, 0.0, 0.0, 0.0]), 
            # #                 euler2quat([np.pi/4,0,0])))
            # start_quat = list(quatMult(
            #                     euler2quat([np.pi/4,0,0]),
            #                     euler2quat([0, 2*np.pi/3,0]), 
            #                     )
            #                 )
            # start_pose = start_pos + start_quat
            # self.pc.goto_pose(start_pose, velocity=0.1)

            d = input("============ Press h to pitch high, m for medium, l for low, press a to abort...")
            if d == 'a':
                break
            if d == 'h':
                p_op = 'high'
            elif d == 'm':
                p_op = 'med'
            elif d == 'l':
                p_op = 'low'
            else:
                raise TypeError

            d = input("============ Press h to xdot high, l for xdot low, press a to abort...")
            if d == 'a':
                break
            if d == 'h':
                x_op = 'high'
            elif d == 'l':
                x_op = 'low'
            else:
                raise TypeError

            d = input("============ Press h to pdot high, l for pdot low, press a to abort...")
            if d == 'a':
                break
            if d == 'h':
                pdot_op = 'high'
            elif d == 'l':
                pdot_op = 'low'
            else:
                raise TypeError

            # spline            
            if p_op == 'high':
                p_v = 10
            elif p_op == 'low':
                p_v = 2
            elif p_op == 'med':
                p_v = 5
            if x_op == 'high':
                x_v = np.array([0.6, 0.7])
            elif x_op == 'low':
                x_v = np.array([0.4, 0.5])
            if pdot_op == 'high':
                pdot_v = np.array([-0.5, -0.5])
            elif pdot_op == 'low':
                pdot_v = np.array([-0.3, -0.3])


            action = np.array([0, 
                            p_v, 
                            -0.05, 
                            0.6, 0.7, 0.2,
                            #    0.4, 0.6, 0.2,   # slower
                                -0.2, -0.5, -0.5])
            action[3:5] = x_v
            action[7:9] = pdot_v

            # Time for spline 
            tn = [0, 0.25, 0.1, 0.25, 0.1, 0.1]
            tn = np.cumsum(tn)
            t_total = tn[-1]
            # ts = np.arange(0, tn[-1], self.dt)

            # Spline for x direction
            xd = np.zeros((6))
            xd[1:4] = action[3:6]
            s_xd = scipy.interpolate.CubicSpline(tn, xd, bc_type='clamped')
            # xds = s_xd(ts)

            # Spline for pitch direction
            pitchd = np.zeros((6))
            pitchd[2:5] = action[6:9]
            s_pitchd = scipy.interpolate.CubicSpline(tn, pitchd, bc_type='clamped')
            # pitchds = s_pitchd(ts)

            # Spline for z direction
            zd = [0, -0.01, 0.02, 0.02, 0, 0]
            s_zd = scipy.interpolate.CubicSpline(tn, zd, bc_type='clamped')
            # zds = s_zd(ts)

            d = input("============ Press Enter to move to initial pose, press a to abort...")
            if p_op == 'high':
                init_joint_angles = [ 7.40926920e-05 , 1.23696712e-01, -7.16082993e-05, -2.52527478e+00, 1.83716418e-05 , 2.64896635e+00 , 7.85783243e-01] # panda, 0, 10, -0.05
            elif p_op == 'med':
                init_joint_angles = [ 9.78801077e-05 , 1.51212684e-01 ,-9.29708437e-05, -2.54376963e+00, 3.26353254e-05 , 2.69497723e+00,  7.85772706e-01]	# panda, 0, 5, -0.05
            elif p_op == 'low':
                init_joint_angles = [ 1.13685017e-04,  1.70841365e-01 ,-1.06591705e-04, -2.55177112e+00, 4.42625313e-05,  2.72260716e+00  ,7.85764661e-01] # panda, 0, 2, -0.05
            self.pc.goto_joints(init_joint_angles, velocity=0.2)

            # Grasp
            d = input("============ Press Enter to grip, press a to abort...")
            if d == 'a':
                break
            self.pc.grasp(width=0.0, e_inner=0.1, e_outer=0.1, speed=0.05, 
                force=80)
            time.sleep(1)

            # Switch
            d = input("============ Press Enter to switch to velocity control, press a to abort...")
            if d == 'a':
                break
            self.cs.switch_controller('velocity')
            time.sleep(0.1)
            rate = rospy.Rate(self.curr_velocity_publish_rate)
            for _ in range(10):
                rate.sleep()
            # settle?
            self.panda.q = np.array(self.joint_state.position[:7])
            for _ in range(100):
                d = self.panda.jacob0()

            # Record
            dq_d_all = []
            dq_all = []
            ddq_all = []
            dddq_all = []
            vel_d_all = []
            vel_all = []
            dq_prev = np.zeros((7))
            ddq_prev = np.zeros((7))

            # Time
            t_init = rospy.get_rostime().to_sec()
            t_cur = 0
            t_prev = -0.002 # arbitrary

            # Segment
            while t_cur <= t_total:

                # Get current joint angles
                self.panda.q = np.array(self.joint_state.position[:7])
                dq = np.array(self.joint_state.velocity[:7])
                jac = self.panda.jacob0()
                v = jac.dot(dq)

                # Find target vel
                t_cur = rospy.get_rostime().to_sec() - t_init
                rospy.loginfo(t_cur)
                v_d = [s_xd(t_cur), 0, s_zd(t_cur), 0, s_pitchd(t_cur), 0]

                # # Option 1: pseudoinverse
                # dq_d = np.matmul(np.linalg.pinv(jac), v_d)

                # Option 2: damped
                damping = np.eye((6))*0.002
                pinv = jac.T.dot(np.linalg.inv(jac.dot(jac.T) + damping))
                dq_d = pinv.dot(v_d)

                # Record
                dq_all += [dq]
                dq_d_all += [dq_d]
                ddq = (dq - dq_prev)/(t_cur-t_prev)
                ddq_all += [ddq]
                dddq = (ddq - ddq_prev)/(t_cur-t_prev)
                dddq_all += [dddq]
                vel_d_all += [v_d]
                vel_all += [v]

                dq_prev = dq
                ddq_prev = ddq
                t_prev = t_cur

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

            d = input("============ Press Enter to move down and release spatula, press a to abort...")
            if d == 'a':
                return
            # start_pos = [0.40, 0.0, 0.23]
            # start_quat = list(quatMult(np.array([1.0, 0.0, 0.0, 0.0]), 
            #                 euler2quat([np.pi/4,0,0])))
            # start_pose = start_pos + start_quat
            release_joints = [-1.31958552e-04,  6.14211210e-01,  1.89370051e-04 ,-2.16710011e+00, 2.26838374e-04,  3.56670050e+00,  7.85528469e-01] # [0.60, 0, 0.20]
            self.pc.goto_joints(release_joints, velocity=0.2)
            # self.pc.goto_pose(start_pose, velocity=0.1)

            # Open gripper
            self.pc.set_gripper(width=0.1)
            time.sleep(0.1)

            # Back to home
            d = input("============ Press Enter to home, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(start_joint_angles, velocity=0.2)
            # self.pc.goto_pose(start_pose, velocity=0.1)

        # Release after abort
        self.pc.set_gripper(width=0.1)
        rospy.sleep(0.1)


if __name__ == '__main__':
    scoop_env = ScoopEnv()
    scoop_env.go()
