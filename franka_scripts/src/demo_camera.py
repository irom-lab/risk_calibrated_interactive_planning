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
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from franka_irom.srv import GraspInfer2, GraspInfer2Response


cv_bridge = CvBridge()


class CameraEnv(object):
    def __init__(self):
        super(CameraEnv, self).__init__()
        self.rgb_img = None
        
        # block and bowl attr
        self.block_color = None
        self.bowl_color = None
        #  history
        self.block_past = None
        self.bowl_past = None
        

        # xyz limits for point cloud
        self.pc_x_min = 0.2
        self.pc_x_max = 0.8
        self.pc_y_min = -0.3
        self.pc_y_max = 0.3
        self.pc_z_min = 0.
        self.pc_z_max = 0.5

        self.pub2 = rospy.Publisher('/camera_flag', Bool, queue_size=1)

        # Subscribe to camera
        rospy.Subscriber('/rgb_to_depth/image_raw', Image, self.__rgb_callback)
        rospy.Subscriber('/depth/image_raw', Image, self.__depth_callback)
        rospy.Subscriber('depth/camera_info', CameraInfo, self.__camera_info_callback)
        # Subscribe to user_input
        rospy.Subscriber('/color_strings', String, self.__color_callback)


        # Static transform from world to depth - for transforming the point cloud into world frame
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        self.world_depth_trans = tf_buffer.lookup_transform(
                                                'panda_link0',
                                                'depth_camera_link',
                                                rospy.Time(0),
                                                rospy.Duration(1.0))
        

        # Set up service
        obj_srv = rospy.Service('get_objs_pos', GraspInfer2, self.get_objs_pos)

        

    def __color_callback(self, msg):
        # Process the received color string
        color_string = msg.data
        colors = color_string.split() # Assuming colors are space-separated
        if len(colors) == 2:
            self.block_color, self.bowl_color = colors
        


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


    def get_objs_pos(self, req):
       # print('service called')
        res = GraspInfer2Response() 
        res.yaw = 0 # TODO: use object yaw
        point_list = []
        assert self.rgb_img is not None

        # Get centroid
        coords = self.get_centroids(self.rgb_img)
        
        # print(coords)
        if len(coords)==0:
            res.pos1 = Vector3(x=0, y=0, z=0)
            res.pos2 = Vector3(x=0, y=0, z=0) # might need to change 
        
            return res
        

        # Debug: show centroid
        # plt.imshow(self.rgb_img)
        # plt.scatter(px, py, c='r', s=10)
        # plt.show()

        # Convert raw values to millimeter
        cv_img = cv_bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding='16UC1') # (576,640)
    
        # Get raw depth value at the pixel
        print('check_point')
        raw_depth_value_block = cv_img[coords[1], coords[0]]
        raw_depth_value_bowl = cv_img[coords[3], coords[2]]
        
        depth_values = [raw_depth_value_block,raw_depth_value_bowl]

        # Convert depth image to point cloud
        points = self.convert_depth_to_pc(cv_img)

        
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'depth_camera_link'

        
        ind = 0
        for raw_depth_value in depth_values:
            # Make a PointCloud for a single pixel
            Z = raw_depth_value*0.001
            x = (coords[2*ind]-self.cx)*Z*self.fx
            y = (coords[2*ind+1]-self.cy)*Z*self.fy
            cloud_points = [[x,y,Z]]
            data = pc2.create_cloud_xyz32(header, cloud_points)

             # Transform PointCloud
            data = do_transform_cloud(data, self.world_depth_trans)

             # Convert to points
            point = np.array([pp for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))])[0]
            print('check_point_2')
            #Append points to point list
            point_list.append(point)
            ind += 1
            print('check_point_3')
        res.pos1 = Vector3(x=point_list[0][0], y=point_list[0][1], z=point_list[0][2])
        res.pos2 = Vector3(x=point_list[1][0], y=point_list[1][1], z=point_list[1][2])
            
        return res





    def get_centroids(self, img):
        
        # dictionary of lower and upper bounds for used colors
        # run color_calibration first to ensure visible contours within range
        color_dict_HSV = {
              'green_bowl': [np.array([59, 70, 60]), np.array([76, 150, 85])],
              'blue_bowl': [np.array([100, 50, 200]), np.array([105, 145, 240])],
                'pink_bowl': [np.array([150, 20, 170]), np.array([180, 95, 255])],
              'yellow_block': [np.array([29, 60, 200]),np.array([33, 180, 255])],
              'green_block': [np.array([40, 60, 140]), np.array([75, 120, 220])],
                'red_block': [np.array([6, 210, 160]), np.array([9, 245, 210])],
                'orange_block': [np.array([14, 150, 200]), np.array([17, 210, 250])]
              }
        # coordinate list of block and bowl
        coordinates = []

        if self.block_color == self.block_past and self.bowl_color == self.bowl_past:
            return coordinates
        else:
            self.block_past = self.block_color
            self.bowl_past = self.bowl_color
        
        hsv_bounds_block = color_dict_HSV[self.block_color+'_block']
        hsv_bounds_bowl = color_dict_HSV[self.bowl_color+'_bowl']
        print('block',self.block_color)
        print('bowl',self.bowl_color)

        

        # Convert RGB to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)



        # Threshold the HSV image to get only the desired color
        mask_block = cv2.inRange(hsv, hsv_bounds_block[0], hsv_bounds_block[1])
        mask_bowl = cv2.inRange(hsv, hsv_bounds_bowl[0], hsv_bounds_bowl[1])

        mask_list = [mask_block, mask_bowl]

        fig, axes = plt.subplots(1, 2)
        idd = 0
        for mks in mask_list:
            axes[idd].imshow(cv2.bitwise_and(img, img, mask=mks))
            idd+=1
        plt.show()
        
        
       
        
        max_cnt = 1
        show_img = img.copy()
        color_list=[(0,255,0),(255,0,0)]
# compute the minimum enclosing circles and centroids for block and bowl contours
        idx = 0
        for mks in mask_list:
            
            cnts = cv2.findContours(mks, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_cnt]
            for c in cnts:
                ((px, py), radius) = cv2.minEnclosingCircle(c)
                px = int(px)
                py = int(py)
                coordinates.extend([px, py]) # x1, y1, x2, y2
                M=cv2.moments(c)
                if M["m00"] !=0: # image moments for display purposes
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(show_img, "XX", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(show_img, (px, py), int(radius),
            color_list[idx], 2)
            idx+=1 
        
        # cv2.startWindowThread()
        # cv2.imshow("circled:Press a key to close",show_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return coordinates # 0,1 -> block coordiantes 2,3 -> bowl coordinates

        

        
  



    def infer(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            r.sleep()


if __name__ == '__main__':
    rospy.init_node('demo_camera_env',anonymous=True)
    camera_env = CameraEnv()
    camera_env.infer()
