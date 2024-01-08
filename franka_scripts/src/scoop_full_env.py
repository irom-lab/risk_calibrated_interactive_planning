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
        rospy.Subscriber('/franka_state_controller/F_ext', 
                        WrenchStamped, 
                        self.__F_ext_callback, 
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
        q_pre_grasp_top = [-0.19557871, 0.34991918, -0.13038322, -1.70913879, -0.10698297, 1.62541058, 0.46619168]
        q_pre_grasp = [-0.20530168, 0.34071839, -0.14335436, -1.97265637, -0.11893242, 1.88089919, 0.47500051]
        q_pre_grasp_back = [-0.19632091, 0.24706826, -0.15283009, -1.94550678, -0.1267923, 1.76122394, 0.46334469]
        # q_pre_scoop = [-0.01649024, 0.17897853, 0.01583383, -2.48667121, -0.00559315, 2.58326533, 0.78892188]  # 5 deg pitch
        # q_pre_scoop_up = [-0.02082344, 0.13152713, 0.02035384, -2.48069537, -0.00486302, 2.52983497, 0.78840998]
        q_pre_scoop = [-0.01749057, 0.1831023, 0.01716866, -2.40309851, -0.00492119, 2.41654679, 0.78846331]
        q_pre_scoop_up = [-0.0203183, 0.14576929, 0.02015296, -2.38775959, -0.00431673, 2.3638737, 0.78804847]
        q_pre_dump = [0.20704281, 0.3029123, 0.12024186, -2.00613774, -0.05068853, 2.31099573, 1.14074275]
        q_dump = [0.47337698, 0.39329426, -0.06361142, -1.83626968, -0.70402573, 1.79216577, 1.51512498]
        # q_pre_grasp_final = [-0.11345606, 0.34366243, -0.24196185, -1.97652853, -0.08794587, 1.88414526, 0.45084214]
        # q_pre_grasp_back_final = [-0.05360313, 0.25217829, -0.30584088, -1.94929097, -0.09406855, 1.76489741, 0.43638302]
        q_pre_grasp_final = [-0.10753292, 0.33234923, -0.2498266, -1.98921702, -0.08910607, 1.88553225, 0.44912848]
        q_pre_grasp_back_final = [-0.04714365, 0.24064747, -0.31392275, -1.96106002, -0.09644238, 1.76529135, 0.43547033]

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

            d = input("============ Press Enter to move to pre grasp top, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp_top, velocity=0.3)

            d = input("============ Press Enter to move to pre grasp, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp, velocity=0.3)

            d = input("============ Press Enter to grip, press a to abort...")
            if d == 'a':
                break
            self.pc.grasp(width=0.0, e_inner=0.1, e_outer=0.1, speed=0.05, 
                force=80)
            time.sleep(0.1)

            d = input("============ Press Enter to move to pre grasp back, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp_back, velocity=0.3)

            d = input("============ Press Enter to move to pre scoop up, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_scoop_up, velocity=0.3)

            d = input("============ Press Enter to move to pre scoop, press a to abort...")
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
            t_total = 2.3 # spatula tip starts at 2cm above table
            t_init = rospy.get_rostime().to_sec()
            wrench_all = []
            while t_cur <= t_total:

                # Get current joint angles
                self.panda.q = np.array(self.joint_state.position[:7])
                dq = np.array(self.joint_state.velocity[:7])
                jac = self.panda.jacob0()
                v_d = [0, 0, -0.01, 0, 0, 0]
                t_cur = rospy.get_rostime().to_sec() - t_init
                wrench_all += [self.F_ext.wrench]

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

            # Plot
            # force_x = [w.force.x for w in wrench_all]
            # force_y = [w.force.y for w in wrench_all]
            # force_z = [w.force.z for w in wrench_all]
            # torque_x = [w.torque.x for w in wrench_all]
            # torque_y = [w.torque.y for w in wrench_all]
            # torque_z = [w.torque.z for w in wrench_all]
            # import matplotlib.pyplot as plt
            # plt.rcParams['font.size'] = 18
            # # plt.plot(force_x, linewidth=5, label='force x')
            # # plt.plot(force_y, linewidth=5, label='force_y')
            # plt.plot(force_z, linewidth=3, label='force_z')
            # # plt.plot(torque_x, linewidth=5, label='torque_x')
            # # plt.plot(torque_y, linewidth=5, label='torque_y')
            # # plt.plot(torque_z, linewidth=5, label='torque_z')
            # plt.legend()
            # plt.show()

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
                x_v = np.array([0.4, 0.4])
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
            s_xd = scipy.interpolate.CubicSpline(tn, xd, bc_type='clamped')
            # xds = s_xd(ts)

            # Spline for pitch direction
            pitchd = np.zeros((6))
            pitchd[2:5] = action[6:9]
            s_pitchd = scipy.interpolate.CubicSpline(tn, pitchd, bc_type='clamped')
            # pitchds = s_pitchd(ts)

            # Spline for z direction
            zd = [0, 0, 0.01, 0.02, 0, 0]   # was -0.01 for 2nd
            s_zd = scipy.interpolate.CubicSpline(tn, zd, bc_type='clamped')
            # zds = s_zd(ts)

            # Switch
            d = input("============ Press Enter to switch to velocity control, press a to abort...")
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
                wrench_all += [self.F_ext.wrench]

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

            # Plot force and wrench
            force_x = [w.force.x for w in wrench_all]
            force_y = [w.force.y for w in wrench_all]
            force_z = [w.force.z for w in wrench_all]
            torque_x = [w.torque.x for w in wrench_all]
            torque_y = [w.torque.y for w in wrench_all]
            torque_z = [w.torque.z for w in wrench_all]
            import matplotlib.pyplot as plt
            plt.rcParams['font.size'] = 18
            f, axarr = plt.subplots(1,2)
            axarr[0].plot(force_x, linewidth=3, label='force x')
            axarr[0].plot(force_y, linewidth=3, label='force_y')
            axarr[0].plot(force_z, linewidth=3, label='force_z')
            axarr[1].plot(torque_x, linewidth=3, label='torque_x')
            axarr[1].plot(torque_y, linewidth=3, label='torque_y')
            axarr[1].plot(torque_z, linewidth=3, label='torque_z')
            axarr[0].legend(prop={'size': 30})
            axarr[1].legend(prop={'size': 30})
            plt.show()

            d = input("============ Press Enter to move to pre dump top, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_dump, velocity=0.3)

            d = input("============ Press Enter to dump, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_dump, velocity=0.3)

            d = input("============ Press Enter to move to pre grasp back, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp_back_final, velocity=0.3)

            d = input("============ Press Enter to pre grasp, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp_final, velocity=0.3)

            d = input("========= Press Enter to open gripper, press a to abort...")
            if d == 'a':
                break
            self.pc.set_gripper(width=0.1)
            time.sleep(0.1)

            d = input("============ Press Enter to move to pre grasp top, press a to abort...")
            if d == 'a':
                break
            self.pc.goto_joints(q_pre_grasp_top, velocity=0.3)

            # # Back to home
            # d = input("============ Press Enter to home, press a to abort...")
            # if d == 'a':
            #     break
            # self.pc.goto_joints(q_init, velocity=0.2)
            # # self.pc.goto_pose(start_pose, velocity=0.1)

        # Open gripper
        d = input("========= Finishing... Press Enter to open gripper, press a if not...")
        if d == 'a':
            return
        self.pc.set_gripper(width=0.1)
        time.sleep(0.1)


if __name__ == '__main__':
    scoop_env = ScoopEnv()
    scoop_env.go()
