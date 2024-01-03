#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import rospy
import random
import tf2_ros
import tf2_py as tf2
import cv2
import std_msgs
from sensor_msgs.msg import Image, CameraInfo

# from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Vector3
import sensor_msgs.point_cloud2 as pc2
from franka_irom.srv import Pos, PosResponse 
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from cv_bridge import CvBridge, CvBridgeError
import ros_numpy

cv_bridge = CvBridge()


class PushCameraEnv(object):
    """For inferring bottle location using calibrated K4A and point cloud"""

    def __init__(self):
        super(PushCameraEnv, self).__init__()
        self.pc_x_min = 0.55
        self.pc_x_max = 1.1
        self.pc_y_min = -0.3
        self.pc_y_max = 0.3
        self.pc_z_min = 0.02
        self.pc_z_max = 0.3

        # Static transform from world to depth - for transforming the point cloud into world frame
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        self.world_depth_trans = tf_buffer.lookup_transform(
                                                'panda_link0',
                                                'depth_camera_link',
                                                rospy.Time(0),
                                                rospy.Duration(1.0))

        # Subscribe to camera info
        rospy.Subscriber('depth/camera_info', 
                        CameraInfo,
                        self.__camera_info_callback, 
                        queue_size=1)

        # Subscribe to depth
        rospy.Subscriber('/depth/image_raw', Image, self.__depth_callback)

        # # Subscribe to point cloud
        # rospy.Subscriber('/points2', 
        #                 PointCloud2, 
        #                 self.__point_cloud_callback, 
        #                 queue_size=1)

        # Service for reporting bottle position
        self.pos_pub = rospy.Service('~get_bottle_pos', Pos, self.get_bottle_pos)


    def __depth_callback(self, msg):
        self.depth_msg = msg
        # if self.flag_recording and self.bag is not None:
            # self.bag.write('depth', msg, msg.header.stamp)

    # def __point_cloud_callback(self, data):
    #     data = do_transform_cloud(data, self.world_depth_trans)
    #     self.pc_points = np.array([pp for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z")) \
    #         if pp[2] < self.pc_z_max and pp[2] > self.pc_z_min and \
    #            pp[0] < self.pc_x_max and pp[0] > self.pc_x_min and \
    #            pp[1] < self.pc_y_max and pp[1] > self.pc_y_min])
        # pc = ros_numpy.numpify(data)
        # points = np.zeros((pc.shape[0],3))
        # points[:,0] = pc['x']
        # points[:,1] = pc['y']
        # points[:,2] = pc['z']
        # p = pcl.PointCloud(np.array(points, dtype=np.float32))


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
        points = np.array([pp for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z")) \
            if pp[2] < self.pc_z_max and pp[2] > self.pc_z_min and \
               pp[0] < self.pc_x_max and pp[0] > self.pc_x_min and \
               pp[1] < self.pc_y_max and pp[1] > self.pc_y_min
            ])
        return points


    def get_bottle_pos(self, req):
        res = PosResponse()
        # res.pos.x = 0
        # res.pos.y = 0
        # res.pos.z = 0

        # Convert raw values to millimeter
        cv_img = cv_bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding='16UC1') # (576,640)
        points = self.convert_depth_to_pc(cv_img)

        # points = self.pc_points[np.random.choice(self.pc_points.shape[0], 300, replace=False)]
        # points = self.pc_points
        pos_avg = np.mean(points, axis=0)
        res.pos.x = pos_avg[0]
        res.pos.y = pos_avg[1]
        res.pos.z = pos_avg[2]
        # print(points.shape)
        # from mpl_toolkits import mplot3d
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(points[:,0], points[:,1], points[:,2], c = 'b', marker='o')
        # ax.scatter(pos_avg[0], pos_avg[1], pos_avg[2], c = 'k', marker='o')
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')
        # plt.show()

        return res


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    rospy.init_node('push_camera_env')
    push_camera_env = PushCameraEnv()
    rospy.spin()
