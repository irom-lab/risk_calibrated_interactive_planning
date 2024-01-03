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

import rosbag
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

        # Initialize rospy node
        rospy.init_node('convert_env', anonymous=True)

        # Publish rosbag msg for conversion
        self.rgb_publisher = rospy.Publisher(
                            'rgb_for_conversion', 
                            Image, 
                            queue_size=5)

        # # Publish rosbag msg for conversion
        # self.camera_info_publisher = rospy.Publisher(
        #                     '/camera_info_for_conversion', 
        #                     CameraInfo, 
        #                     queue_size=5)

        # Subscribe to converted points
        rospy.Subscriber('depth_registered/image_rect', 
                        Image,
                        self.__img_callback, 
                        queue_size=5)

        # Subscribe to camera info
        # rospy.Subscriber('depth/camera_info', 
        #                 CameraInfo,
        #                 self.__camera_info_callback, 
        #                 queue_size=1)


    def __img_callback(self, data):
        # Convert image to numpy array
        self.img_converted = cv_bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
        # img = np.array(img, dtype=np.float32)
        # img = np.array(img, dtype=np.uint16)
        # img = np.array(img, dtype=np.uint8)

    # def __camera_info_callback(self, data):
    #     self.cx = data.K[2]
    #     self.cy = data.K[5]
    #     self.fx = 1/data.K[0]
    #     self.fy = 1/data.K[4]


    # def convert_depth_to_pc(self, img):
    #     # Make a PointCloud first
    #     header = std_msgs.msg.Header()
    #     header.stamp = rospy.Time.now()
    #     header.frame_id = 'depth_camera_link'
    #     cloud_points = []
    #     cnt = 0
    #     for v in range(img.shape[0]):
    #         for u in range(img.shape[1]):
    #             if img[v,u] > 0:
    #                 Z = img[v,u]*0.001
    #                 x = (u-self.cx)*Z*self.fx
    #                 y = (v-self.cy)*Z*self.fy
    #                 cloud_points += [[x,y,Z]]
    #                 cnt += 1
    #     data = pc2.create_cloud_xyz32(header, cloud_points)

    #     # Transform PointCloud
    #     data = do_transform_cloud(data, self.world_depth_trans)

    #     # Filter points
    #     points = np.array([pp for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z")) \
    #         if pp[2] < self.pc_z_max and pp[2] > self.pc_z_min and \
    #            pp[0] < self.pc_x_max and pp[0] > self.pc_x_min and \
    #            pp[1] < self.pc_y_max and pp[1] > self.pc_y_min
    #         ])
    #     return points


    def run(self):
        video_folder_prefix = sys.argv[1]
        bag_path_list = glob.glob(os.path.join(video_folder_prefix, '*_k4a.bag'))
        save_path = os.path.join(video_folder_prefix, 'pos_traj.pkl')
        rate = rospy.Rate(10)
        for _ in range(10):
            rate.sleep()

        self.waiting_for_callback = True

        pos_traj_all = []

        # Process each file
        for i, bag_path in enumerate(bag_path_list):
            print(f'Processing {bag_path}...')

            # Load bag
            bag = rosbag.Bag(bag_path)

            # Read frames
            # min_frame = 40
            # max_frame = 80
            # min_num_frame = 5
            # starting_pos_norm_diff_threshold = 0.008
            # ending_pos_norm_diff_threshold = 0.008
            cnt = 0
            # pos_traj = []
            # bottle_moving = False
            # prev_norm_diff = deque(maxlen=5)
            for topic, msg, t in bag.read_messages(topics=['image']):
                print('Frame: ', cnt)
                cnt += 1
                # if cnt < min_frame or cnt > max_frame:
                #     continue


                # Wait for callback
                while self.waiting_for_callback:

                    # Publish msg
                    self.rgb_publisher.publish(msg)

                    rate.sleep()
                
                # Convert raw values to millimeter
                cv_img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8') # (576,640)

                # Convert depth image to point cloud
                # points = self.convert_depth_to_pc(cv_img)

                # # Find bottle front - assume no object further in x
                # front_ind = np.argmax(points[:,0])
                # x_lower = points[front_ind, 0] - 0.07   # 0.06m diameter of the green bottle
                # y_lower = points[front_ind, 1] - 0.033
                # y_upper = points[front_ind, 1] + 0.033

                # # Filter the arm
                # points_after = points[np.logical_and(np.logical_and(points[:,0]>x_lower, points[:,1]>y_lower), points[:,1]<y_upper)]

                # # Get pos
                # pos = np.mean(points_after, axis=0)

                # # Record initial pos
                # if cnt == min_frame:
                #     pos_traj += [pos[None]]
                #     prev_pos = pos

                # # Determine if bottle is moving
                # pos_norm_diff = np.linalg.norm(pos[:2]-prev_pos[:2])
                # print('Pos: ', pos)
                # print('Norm diff: ', pos_norm_diff)
                # prev_pos = pos

                # if pos_norm_diff > starting_pos_norm_diff_threshold and not bottle_moving:
                #     bottle_moving = True
                #     start_frame = cnt
                #     print('start!')
                # if bottle_moving:
                #     pos_traj += [pos[None]]
                # if pos_norm_diff < ending_pos_norm_diff_threshold and bottle_moving and cnt-start_frame > min_num_frame:
                #     bottle_moving = False
                #     print('end!')
                #     break

                # Debug
                # if bottle_moving and pos[0] > 0.8:
                #     plt.imshow(cv_img)
                #     plt.show()

                #     from mpl_toolkits import mplot3d
                #     fig = plt.figure()
                #     ax = fig.add_subplot(111, projection='3d')
                #     ax.scatter(points[:,0], points[:,1], points[:,2], c='b', marker='o')
                #     ax.set_xlabel('X-axis')
                #     ax.set_ylabel('Y-axis')
                #     ax.set_zlabel('Z-axis')
                #     plt.show()

                #     from mpl_toolkits import mplot3d
                #     fig = plt.figure()
                #     ax = fig.add_subplot(111, projection='3d')
                #     ax.scatter(points_after[:,0], points_after[:,1], points_after[:,2], c='b', marker='o')
                #     ax.set_xlabel('X-axis')
                #     ax.set_ylabel('Y-axis')
                #     ax.set_zlabel('Z-axis')
                #     plt.show()

            # Plot trajectory
            pos_traj = np.vstack(pos_traj)
            fig = plt.figure()
            plt.plot(pos_traj[:,0], c='r')
            plt.plot(pos_traj[:,1], c='g')
            plt.plot(pos_traj[:,2], c='b')
            plt.legend()
            plt.show()

            # Save traj
            pos_traj_all += [pos_traj]
            print('Number of frames in the trajectory: ', len(pos_traj))
            bag.close()

        # save to pickle file
        with open(save_path, 'wb') as f:
            pickle.dump(pos_traj_all, f)


if __name__ == '__main__':
    convert_env = ConvertEnv()
    convert_env.run()
