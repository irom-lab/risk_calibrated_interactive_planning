#!/usr/bin/env python3
import os
import sys
import numpy as np
import rospy
import glob
import matplotlib.pyplot as plt
import tf2_ros
import tf2_py as tf2
import pickle

import cv2
from cv_bridge import CvBridge, CvBridgeError
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import std_msgs
from sensor_msgs.msg import Image, CameraInfo
# from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

cv_bridge = CvBridge()


class ConvertEnv(object):
    def __init__(self):
        super(ConvertEnv, self).__init__()

        # Constants
        self.pc_x_min = 0.70
        self.pc_x_max = 0.85
        self.pc_y_min = -0.10
        self.pc_y_max = 0.10
        self.pc_z_min = -0.05
        self.pc_z_max = 0.20
        # self.pc_x_min = -10
        # self.pc_x_max = 10
        # self.pc_y_min = -10
        # self.pc_y_max = 10
        # self.pc_z_min = -10
        # self.pc_z_max = 10

        # Initialize rospy node
        rospy.init_node('convert_env', anonymous=True)

        # Static transform from world to depth - for transforming the point cloud into world frame
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        self.world_depth_trans = tf_buffer.lookup_transform(
                                                'panda_link0',
                                                'depth_camera_link',
                                                rospy.Time(0),
                                                rospy.Duration(1.0))

        # Node is subscribing to the depth topic
        rospy.Subscriber('/depth/image_raw', Image, self.__depth_callback)

        # Subscribe to camera info
        rospy.Subscriber('depth/camera_info', 
                        CameraInfo,
                        self.__camera_info_callback, 
                        queue_size=1)


    def __depth_callback(self, msg):
        self.msg = msg


    def __camera_info_callback(self, data):
        self.cx = data.K[2]
        self.cy = data.K[5]
        self.fx = 1/data.K[0]
        self.fy = 1/data.K[4]


    def convert_depth_to_pc(self, img):
        # Make a PointCloud first
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'depth_camera_link'
        cloud_points = []
        # uv_coords = []
        cnt = 0
        for v in range(img.shape[0]):
            for u in range(img.shape[1]):
                if img[v,u] > 0:
                    Z = img[v,u]*0.001
                    x = (u-self.cx)*Z*self.fx
                    y = (v-self.cy)*Z*self.fy
                    cloud_points += [[x,y,Z]]
                    # uv_coords += [[u,v]]
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


    def run(self):
        # video_folder_prefix = sys.argv[1]
        # save_path = os.path.join(video_folder_prefix, 'pos_traj.pkl')
        rate = rospy.Rate(10)
        for _ in range(10):
            rate.sleep()

        # Convert raw values to millimeter
        cv_img = cv_bridge.imgmsg_to_cv2(self.msg, desired_encoding='16UC1') # (576,640)

        # Convert depth image to point cloud
        points = self.convert_depth_to_pc(cv_img)

        # Debug: depth image
        plt.imshow(cv_img)
        plt.show()

        # Debug: point cloud
        from mpl_toolkits import mplot3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        import random
        points_ind = random.sample(range(len(points)), 500)
        points = points[points_ind]
        ax.scatter(points[:,0], points[:,1], points[:,2], c='b', marker='o')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()

        # save to pickle file
        save_path = '/home/allen/catkin_ws/src/franka_irom/data/empth_table_depth.npz'
        out = {'depth_cv2': cv_img,
               'cx': self.cx, 
               'cy': self.cy,
               'fx': self.fx,
               'fy': self.fy}
        with open(save_path, 'wb') as f:
            pickle.dump(out, f)


if __name__ == '__main__':
    convert_env = ConvertEnv()
    convert_env.run()
