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
from util.transform import quatMult, euler2quat, log_rot, quat2list
from util.trajectory import QuinticTimeScaling, LinearTimeScaling

from std_msgs.msg import Float64MultiArray
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped


from pydrake.all import RigidTransform, RotationMatrix, RollPitchYaw



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
        # rospy.Subscriber('/franka_state_controller/F_ext', 
        #                 WrenchStamped, 
        #                 self.__F_ext_callback, 
        #                 queue_size=10)

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


    def __F_ext_callback(self, msg):
        self.F_ext = msg


    def stop(self):
        msg = Float64MultiArray()
        # msg = Twist()
        msg.data = [0., 0., 0., 0., 0., 0., 0.]
        self.curr_velo_pub.publish(msg)


    def go(self):
        
        # Fixed transform from world to EE pointing down
        R_ee_fixed = RotationMatrix(RollPitchYaw(0, -np.pi, np.pi/2)).multiply(RotationMatrix(RollPitchYaw(0, 0, -np.pi/4)))

        # [0.40, 0.0, 0.40]
        q_init = [-0.084, -0.415, 0.0818, -2.584, 0.041, 2.169, 0.754]

        # Common poses
        q_pre_grasp =  [-0.215, 0.438,-0.134,-1.992,-0.114, 1.997, 0.485]
        q_pre_grasp_top =  [-0.201, 0.41 ,-0.125,-1.771,-0.103, 1.747, 0.478]
        q_pre_grasp_back =  [-0.205, 0.327,-0.146,-1.991,-0.121, 1.885, 0.474]

        # Open gripper
        d = input("========= Press Enter to open gripper, press a to abort...")
        if d == 'a':
            return
        self.pc.set_gripper(width=0.1)
        time.sleep(0.1)

        # Loop scoops
        while not rospy.is_shutdown():

            d = input("============ Press Enter to move to start pose, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_init, velocity=0.3)
            # T_init = [0.40, 0.0, 0.40]
            # R_init = R_ee_fixed
            # Q_init = quat2list(R_init.ToQuaternion())
            # list_p_init = T_init + Q_init
            # self.pc.goto_pose(list_p_init, velocity=0.1)

            while 1:
                d = input("============ Press h to 10 deg pitch, m for 8 deg, l for 5 deg...")
                if d == 'h':
                    q_pre_scoop_up = [-0.019, 0.282, 0.018,-2.39 ,-0.009, 2.502, 0.791]
                    q_pre_scoop = [-0.015, 0.326, 0.014,-2.396,-0.009, 2.552, 0.791]
                    q_pre_dump = [0.217, 0.346, 0.109,-2.02 ,-0.055, 2.368, 1.144]
                    q_dump = [0.59 , 0.542,-0.12 ,-1.754,-0.644, 1.82 , 1.58 ]
                    q_pre_grasp_back_final = [-0.02 , 0.336,-0.344,-1.998,-0.061, 1.891, 0.426]
                    q_pre_grasp_final = [-0.092, 0.442,-0.266,-2.   ,-0.056, 2.002, 0.442]
                    break
                elif d == 'm':
                    q_pre_scoop_up = [-0.018, 0.28 , 0.017,-2.425,-0.009, 2.57 , 0.791]
                    q_pre_scoop = [-0.013, 0.328, 0.012,-2.427,-0.008, 2.62 , 0.791]
                    q_pre_dump = [ 0.219, 0.346, 0.107,-2.02 ,-0.054, 2.368, 1.143]
                    q_dump = [ 0.6  , 0.543,-0.132,-1.755,-0.639, 1.823, 1.577]
                    q_pre_grasp_back_final = [-0.011, 0.337,-0.354,-1.998,-0.057, 1.891, 0.424]
                    q_pre_grasp_final = [-0.083, 0.444,-0.275,-2.   ,-0.052, 2.002, 0.439]
                    break
                elif d == 'l':
                    q_pre_scoop_up = [-0.015, 0.283, 0.014,-2.473,-0.009, 2.673, 0.791]
                    q_pre_scoop = [-0.009, 0.336, 0.008,-2.469,-0.007, 2.723, 0.79 ]
                    q_pre_dump = [ 0.223, 0.346, 0.102,-2.02 ,-0.052, 2.369, 1.142]
                    q_dump = [ 0.616, 0.544,-0.151,-1.756,-0.63 , 1.827, 1.572]
                    q_pre_grasp_back_final = [ 0.004, 0.339,-0.37 ,-1.998,-0.052, 1.892, 0.419]
                    q_pre_grasp_final = [-0.068, 0.445,-0.291,-2. ,-0.045, 2.002, 0.433]
                    break

            d = input("============ Press Enter to pick up spatula, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp_top, velocity=0.3)
            self.pc.goto_joints(q_pre_grasp, velocity=0.3)

            d = input("============ Press Enter to pick up spatula, press a to abort...")
            if d == 'a':
                break
            self.pc.grasp(width=0.0, e_inner=0.1, e_outer=0.1, speed=0.05, 
                force=80)
            time.sleep(0.1)

            d = input("============ Press Enter to move spatula to table, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp_back, velocity=0.3)
            self.pc.goto_joints(q_pre_scoop_up, velocity=0.3)

            d = input("============ Press Enter to move spatula to table, press a to abort...")
            if d == 'a':
                break
            # self.pc.goto_joints(q_pre_scoop, velocity=0.1)
            self.cs.switch_controller('velocity')
            time.sleep(0.1)
            rate = rospy.Rate(self.curr_velocity_publish_rate)
            for _ in range(10):
                rate.sleep()
            self.panda.q = np.array(self.joint_state.position[:7])
            for _ in range(100):
                d = self.panda.jacob0()

            t_cur = 0
            t_total = 1.0 # spatula tip starts at 2cm above table
            t_init = rospy.get_rostime().to_sec()
            wrench_all = []
            while t_cur <= t_total:

                # Get current joint angles
                self.panda.q = np.array(self.joint_state.position[:7])
                dq = np.array(self.joint_state.velocity[:7])
                jac = self.panda.jacob0()
                v_d = [0, 0, -0.02, 0, 0, 0]
                t_cur = rospy.get_rostime().to_sec() - t_init
                # wrench_all += [self.F_ext.wrench]

                # Option 2: damped
                damping = np.eye((6))*0.002
                pinv = jac.T.dot(np.linalg.inv(jac.dot(jac.T) + damping))
                dq_d = pinv.dot(v_d)

                # Record
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

            while 1:
                d = input("============ Press h to xdot high, l for xdot low...")
                if d == 'h':
                    x_op = 'high'
                    break
                elif d == 'l':
                    x_op = 'low'
                    break

            while 1:
                d = input("============ Press h to pdot high, l for pdot low...")
                if d == 'h':
                    pdot_op = 'high'
                    break
                elif d == 'l':
                    pdot_op = 'low'
                    break

            if x_op == 'high':
                x_v = np.array([0.6, 0.7])
            elif x_op == 'low':
                x_v = np.array([0.4, 0.5])
            if pdot_op == 'high':
                pdot_v = np.array([-0.5, -0.6])
            elif pdot_op == 'low':
                pdot_v = np.array([-0.3, -0.4])

            action = np.array([0, 5, -0.05, 
                               0.6, 0.7, 0.2,
                              -0.2, -0.6, -0.6])
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
            # s_xd = scipy.interpolate.CubicSpline(tn, xd, bc_type='clamped')
            s_xd = scipy.interpolate.interp1d(tn, xd)

            # Spline for pitch direction
            pitchd = np.zeros((6))
            pitchd[2:5] = action[6:9]
            # s_pitchd = scipy.interpolate.CubicSpline(tn, pitchd, bc_type='clamped')
            s_pitchd = scipy.interpolate.interp1d(tn, pitchd)

            # Spline for z direction
            zd = [0, 0, 0.01, 0.02, 0, 0]   # was -0.01 for 2nd
            # s_zd = scipy.interpolate.CubicSpline(tn, zd, bc_type='clamped')
            s_zd = scipy.interpolate.interp1d(tn, zd)

            # Switch
            d = input("============ Press Enter to scoop, press a to abort...")
            if d == 'a':
                break
            self.cs.switch_controller('velocity')
            time.sleep(0.1)
            rate = rospy.Rate(self.curr_velocity_publish_rate)
            for _ in range(10):
                rate.sleep()
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
            wrench_all = []

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
                # wrench_all += [self.F_ext.wrench]

                # Find target vel
                t_cur = rospy.get_rostime().to_sec() - t_init
                rospy.loginfo(t_cur)
                if t_cur > t_total:
                    break
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
            dq_d_all = np.vstack(dq_d_all)
            dq_all = np.vstack(dq_all)
            ddq_all = np.vstack(ddq_all)
            dddq_all = np.vstack(dddq_all)
            vel_d_all = np.vstack(vel_d_all)
            vel_all = np.vstack(vel_all)
            import matplotlib.pyplot as plt
            f, axarr = plt.subplots(1,7)
            for joint_ind in range(7):
                axarr[joint_ind].plot(dq_d_all[:,joint_ind], label='desired')
                axarr[joint_ind].plot(dq_all[:,joint_ind], label='actual')
                axarr[joint_ind].legend()
            plt.show()

            f, axarr = plt.subplots(6,1)
            for ind in range(6):
                axarr[ind].plot(vel_d_all[:,ind], label='desired')
                axarr[ind].plot(vel_all[:,ind], label='actual')
                axarr[ind].legend() 
            plt.show()

            d = input("============ Press Enter to dump, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_dump, velocity=0.3)
            self.pc.goto_joints(q_dump, velocity=0.3)

            d = input("============ Press Enter to put spatula back, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp_back_final, velocity=0.3)
            self.pc.goto_joints(q_pre_grasp_final, velocity=0.3)
            self.pc.set_gripper(width=0.1)
            time.sleep(0.1)
            self.pc.goto_joints(q_pre_grasp_top, velocity=0.3)

        # Open gripper
        d = input("========= Finishing demo... Press Enter to open gripper, press a if not...")
        if d == 'a':
            return
        self.pc.set_gripper(width=0.1)
        time.sleep(0.1)


if __name__ == '__main__':
    scoop_env = ScoopEnv()
    scoop_env.go()
