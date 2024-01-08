#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import rospy
from matplotlib import pyplot as plt
import cv2

import tf2_ros
import tf2_py as tf2
import std_msgs
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from franka_irom.srv import GraspInfer, GraspInferResponse


cv_bridge = CvBridge()


class CameraEnv(object):
    def __init__(self):
        super(CameraEnv, self).__init__()
        self.rgb_img = None

        # xyz limits for point cloud
        self.pc_x_min = 0.2
        self.pc_x_max = 0.8
        self.pc_y_min = -0.3
        self.pc_y_max = 0.3
        self.pc_z_min = 0.
        self.pc_z_max = 0.5

        # Subscribe to camera
        rospy.Subscriber('/rgb_to_depth/image_raw', Image, self.__rgb_callback)
        rospy.Subscriber('/depth/image_raw', Image, self.__depth_callback)
        rospy.Subscriber('depth/camera_info', CameraInfo, self.__camera_info_callback)

        # Static transform from world to depth - for transforming the point cloud into world frame
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        self.world_depth_trans = tf_buffer.lookup_transform(
                                                'panda_link0',
                                                'depth_camera_link',
                                                rospy.Time(0),
                                                rospy.Duration(1.0))


        # Set up service
        obj_srv = rospy.Service('get_obj_pos', GraspInfer, self.get_obj_pos)


    def __rgb_callback(self, msg):
        self.rgb_img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def __depth_callback(self, msg):
        self.depth_msg = msg

    def __camera_info_callback(self, data):
        self.cx = data.K[2]
        self.cy = data.K[5]
        self.fx = 1/data.K[0]
        self.fy = 1/data.K[4]

    def convert_depth_to_pc(self, img):
        # Make a PointCloud first - can take a while
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'depth_camera_link'
        cloud_points = []
        cnt = 0
        for v in range(img.shape[0]):
            for u in range(img.shape[1]):
                if img[v,u] > 0:
                    Z = img[v,u]*0.001
                    x = (u-self.cx)*Z*self.fx
                    y = (v-self.cy)*Z*self.fy
                    cloud_points += [[x,y,Z]]
                    cnt += 1
        data = pc2.create_cloud_xyz32(header, cloud_points)

        # Transform PointCloud
        data = do_transform_cloud(data, self.world_depth_trans)

        # Filter points
        points = np.array([pp for pp in pc2.read_points(data, skip_nans=False, field_names=("x", "y", "z"), 
        # uvs=uv_coords
        ) \
            if pp[2] < self.pc_z_max and pp[2] > self.pc_z_min and \
               pp[0] < self.pc_x_max and pp[0] > self.pc_x_min and \
               pp[1] < self.pc_y_max and pp[1] > self.pc_y_min
            ])
        return points

    def get_obj_pos(self, req):
        res = GraspInferResponse()
        res.yaw = 0 # TODO: use object yaw
        assert self.rgb_img is not None

        # Get centroid
        px, py = self.get_centroid(self.rgb_img)
        if px is None:
            res.pos = Vector3(x=0.5, y=0, z=0.1)
            print('No contour found!')
            return res

        # Debug: show centroid
        # plt.imshow(self.rgb_img)
        # plt.scatter(px, py, c='r', s=10)
        # plt.show()

        # Convert raw values to millimeter
        cv_img = cv_bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding='16UC1') # (576,640)
    
        # Get raw depth value at the pixel
        raw_depth_value = cv_img[py, px]

        # Convert depth image to point cloud
        points = self.convert_depth_to_pc(cv_img)

        # Make a PointCloud for a single pixel
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'depth_camera_link'
        Z = raw_depth_value*0.001
        x = (px-self.cx)*Z*self.fx
        y = (py-self.cy)*Z*self.fy
        cloud_points = [[x,y,Z]]
        data = pc2.create_cloud_xyz32(header, cloud_points)

        # Transform PointCloud
        data = do_transform_cloud(data, self.world_depth_trans)

        # Convert to points
        point = np.array([pp for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))])[0]
        res.pos = Vector3(x=point[0], y=point[1], z=point[2])
        return res


    def get_centroid(self, img):

        color_dict_HSV = {
              'green_bowl': [np.array([59, 70, 60]), np.array([76, 150, 85])],
              'blue_bowl': [np.array([100, 50, 200]), np.array([105, 145, 240])],
                'pink_bowl': [np.array([150, 20, 170]), np.array([180, 95, 255])],
              'yellow_block': [np.array([29, 60, 200]),np.array([33, 180, 255])],
              'green_block': [np.array([40, 60, 140]), np.array([75, 120, 220])],
                'red_block': [np.array([6, 210, 160]), np.array([9, 245, 210])],
                'orange_block': [np.array([14, 150, 200]), np.array([17, 210, 250])]
              }
        # use yellow color for now
        # hsv_lower = np.array([20, 100, 230])
       # hsv_upper = np.array([40, 120, 250])
        hsv_lower = np.array([30, 150, 200])
        hsv_upper = np.array([32, 180, 225])

        # test_hsv_lower = np.array([60, 3, 50])
        # test_hsv_upper = np.array([90, 150, 255])

        # Convert RGB to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # # save image for hsv analysis if needed
        save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('/home/allen/catkin_ws/src/franka_irom/src/imageAnalysis.png', save_img)
        print('image saved')

        

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img)
        axes[1].imshow(hsv)
        plt.show()





        # Threshold the HSV image to get only the desired color
        mask = cv2.inRange(hsv, color_dict_HSV['yellow_block'][0], color_dict_HSV['yellow_block'][1])

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(hsv)
        axes[1].imshow(res)
        plt.show()

        mask_ls = [cv2.inRange(hsv, color_dict_HSV[key][0], color_dict_HSV[key][1]) for key in color_dict_HSV.keys() ]



        # mask_list = [mask.copy(),green_bowl_mask.copy(),blue_bowl_mask.copy(),green_block_mask.copy()]
        cnt_list=np.array([])
        color_list = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 255),(0, 0, 0),(255, 255, 0),(0, 255, 255)]
        show_img = img.copy()
        
        #  Prints out all object contoours in mask
        fig, axes = plt.subplots(1, len(color_dict_HSV.keys()))
        idd = 0
        for mks in mask_ls:
            axes[idd].imshow(cv2.bitwise_and(img, img, mask=mks))
            idd+=1
        plt.show()
        # print(cv2.__version__)
        
        idx = 0
        ax,ay = None,None
        for mks in mask_ls:
            cnts = cv2.findContours(mks, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
            for c in cnts:
                ((ax, ay), radius) = cv2.minEnclosingCircle(c)
                ax = int(ax)
                ay = int(ay)
                M=cv2.moments(c)
                if M["m00"] !=0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(show_img, "XX", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(show_img, (ax, ay), int(radius),
            color_list[idx], 2)
            idx+=1
        
        cv2.startWindowThread()
        cv2.imshow("circled:Press a key to close",show_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

            


        # Find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        max_cnts = 1
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_cnts]
        cnt_img = img.copy()

        # compute the minimum enclosing circle and centroid
        px, py = None, None
        i = 0
        for c in cnts:
            ((px, py), radius) = cv2.minEnclosingCircle(c)
            px = int(px)
            py = int(py)
            M = cv2.moments(c)
            # print(px, py, radius)
            # center = (int(M["m10"] / M["m00"]+1e-4), 
            #                 int(M["m01"] / M["m00"]+1e-4))
            # cv2.putText(cnt_img, "Largest", center, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2)
            # print("showing image")
            # plt.imshow(cnt_img)
            # plt.show()
            # try:
            #     i=0
            #     # only proceed if the radius meets a minimum size
            #     if radius+1e-5 > 0:
            #         # draw the circle and centroid on the frame,
            #         # then update the list of tracked points
            #         cv2.putText(cnt_img, "Largest", center, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2)
            #         # cv2.circle(frame, (px, py), int(radius),
            #         #     (0, 255, 255), 2)
            # except:
            #     continue
        return px, py


    def infer(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():

            # Spin
            r.sleep()


if __name__ == '__main__':
    rospy.init_node('camera_env')
    camera_env = CameraEnv()
    camera_env.infer()
