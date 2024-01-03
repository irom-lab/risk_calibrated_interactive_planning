#!/usr/bin/env python3
import argparse
import os
import numpy as np
import rospy
import rosbag
import matplotlib.pyplot as plt
import glob
import random

import cv2
import tf2_ros
import tf2_py as tf2
from cv_bridge import CvBridge, CvBridgeError
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import std_msgs
import sensor_msgs.point_cloud2 as pc2
cv_bridge = CvBridge()


class ConvertEnv(object):
    def __init__(self):
        super(ConvertEnv, self).__init__()

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

        # Load empty table data
        empty_table_data = '/home/allen/catkin_ws/src/franka_irom/data/empth_table_depth.npz'
        empty_table_data = np.load(empty_table_data, allow_pickle=True)
        self.empty_table_depth_cv2 = empty_table_data['depth_cv2']
        self.cx = empty_table_data['cx']
        self.cy = empty_table_data['cy']
        self.fx = empty_table_data['fx']
        self.fy = empty_table_data['fy']

        # Debug: depth image
        # plt.imshow(self.empty_table_depth_cv2)
        # plt.show()


    def get_xy(self, px, py):
        raw_depth_value = self.empty_table_depth_cv2[py, px]
        v = py
        u = px

        # Make a PointCloud for a single pixel
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'depth_camera_link'
        Z = raw_depth_value*0.001
        x = (u-self.cx)*Z*self.fx
        y = (v-self.cy)*Z*self.fy
        cloud_points = [[x,y,Z]]
        data = pc2.create_cloud_xyz32(header, cloud_points)

        # Transform PointCloud
        data = do_transform_cloud(data, self.world_depth_trans)

        # Convert to points
        points = np.array([pp for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))])
        print('Points: ', points)
        return points[0][:2]


    def run(self, video_path_all):
        rate = rospy.Rate(10)
        for _ in range(10):
            rate.sleep()

        # parameters - color range in HSV for contour
        fps = 30
        target_duration = 1
        video_size = (640, 570)
        # hsv_lower = np.array([150,30,10])   # dark chocolate
        # hsv_upper = np.array([190,80,60])
        # hsv_lower = np.array([80,5,230])   # white chocolate
        # hsv_upper = np.array([120,30,255])
        # hsv_lower = np.array([0,180,180]) # carrot
        # hsv_upper = np.array([40,220,220])
        # hsv_lower = np.array([25,15,190]) # cucumber
        # hsv_upper = np.array([55,45,210])
        # hsv_lower = np.array([25,40,150]) # sprouts
        # hsv_upper = np.array([65,120,180])
        # hsv_lower = np.array([0,0,180]) # mushroom
        # hsv_upper = np.array([30,20,220])
        # hsv_lower = np.array([10,110,150]) # pasta
        # hsv_upper = np.array([30,130,170])
        hsv_lower = np.array([110,40,55]) # oreo
        hsv_upper = np.array([130,50,65])
        mid_frame_ratio = 0.5
        late_frame_ratio = 0.75
        num_init_frame = 5  # for averaging

        # Naming convention: all for all trials for this set of experiments, traj for the particular trial

        # Trajectories - should be num x traj_len x 2
        traj_all = []

        # Process each file
        for video_ind, video_path in enumerate(video_path_all):
            print(f'Processing {video_path}...')

            # load video with cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f'Cannot open video file {video_path}!')
                continue
            
            # trajectory of the frame
            traj = []

            # process frames
            frame_cnt = -1
            moving = False
            mid_frame_candidate = []
            late_frame_candidate = []
            final_frame_candidate = []
            frame_traj = []
            hsv_traj = []
            pxy_traj = []
            xy_init_traj = []
            xy_init = None
            prev_frame = None
            thresh_sum_moving_threshold = 50000

            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_cnt += 1
                # print(f'Processing frame: {frame_cnt}...\r', end='')

                # Convert frame from bgr to rgb
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_traj += [frame]

                # Detect motion from previous frame
                if prev_frame is not None:

                    # calculate difference and update previous frame
                    diff_frame = cv2.absdiff(src1=prev_frame, src2=frame)

                    # 5. Only take different areas that are different enough (>20 / 255)
                    thresh_frame = cv2.threshold(src=diff_frame, thresh=200, maxval=255, type=cv2.THRESH_BINARY)[1][:200]
                    thresh_sum = np.sum(thresh_frame)
                    # print()
                    # print(thresh_sum)
                    # print()
                    if thresh_sum > thresh_sum_moving_threshold and not moving:
                        moving = True
                        start_moving_frame = frame_cnt
                        target_end_frame = frame_cnt + target_duration*fps
                        target_mid_frame = int((target_end_frame - start_moving_frame)*mid_frame_ratio) + start_moving_frame
                        target_late_frame = int((target_end_frame - start_moving_frame)*late_frame_ratio) + start_moving_frame
                        print('Start moving at frame: ', start_moving_frame)
                        print('Thres sum: ', np.sum(thresh_frame))

                        # # debug
                        # fig, axes = plt.subplots(1, 2)
                        # axes[0].imshow(frame)
                        # axes[1].imshow(thresh_frame)
                        # plt.show()

                    # if moving:
                    #     fig, axes = plt.subplots(1, 2)
                    #     axes[0].imshow(frame)
                    #     axes[1].imshow(thresh_frame)
                    #     plt.show()

                # # Sharpen frame is moving
                # if moving:
                #     sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                #     frame_sharpen = cv2.filter2D(frame, -1, sharpen_kernel)

                #     # compare
                #     fig, axes = plt.subplots(1, 2)
                #     axes[0].imshow(frame)
                #     axes[1].imshow(frame_sharpen)
                #     plt.show()

                # Convert RGB to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                hsv_traj += [hsv]

                # # Debug: check correct range of the veggie piece in hsv
                # fig, axes = plt.subplots(1, 2)
                # axes[0].imshow(frame)
                # axes[1].imshow(hsv)
                # plt.show()

                # Threshold the HSV image to get only the desired color
                mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

                # Mask out unwanted area - e.g., spatula holder
                if moving and frame_cnt - start_moving_frame > 5:
                    min_py = min(200 + (frame_cnt - start_moving_frame - 5)*10, 
                                 280)
                else:   # not moving
                    min_py = 200
                mask[:min_py, :] = 0
                mask[:, :150] = 0
                mask[:, -150:] = 0
                mask[-150:, :] = 0

                # Bitwise-AND mask and original image
                res = cv2.bitwise_and(frame, frame, mask=mask)

                # show res in matplotlib
                # if frame_cnt >= 75:
                #     fig, axes = plt.subplots(1, 2)
                #     axes[0].imshow(frame)
                #     axes[1].imshow(res)
                #     plt.show()

                # Find the largest two contours in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                max_cnts = 1
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[-2]
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_cnts]
                center = None

                # compute the minimum enclosing circle and centroid
                pxy_frame = []
                xy_frame = []
                radius_frame = []
                for c in cnts:
                    ((px, py), radius) = cv2.minEnclosingCircle(c)
                    px = int(px)
                    py = int(py)
                    M = cv2.moments(c)
                    print(px, py, radius)
                    try:
                        center = (int(M["m10"] / M["m00"]), 
                                  int(M["m01"] / M["m00"]))
                        # only proceed if the radius meets a minimum size
                        if radius > 0:
                            # draw the circle and centroid on the frame,
                            # then update the list of tracked points
                            cv2.circle(frame, (px, py), int(radius),
                                (0, 255, 255), 2)
                            # cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    except:
                        continue
                    pxy_frame.append((px, py))
                    xy = self.get_xy(px, py)
                    xy_frame.append(xy)
                    radius_frame.append(radius)
                print('Pixel XY of all detected contours:', pxy_frame)
                print('XY of all detected contours:', xy_frame)

                # Debug
                # plt.imshow(frame)
                # plt.scatter(px, py, c='r', s=10)
                # plt.show()

                ############# Determine whether to use the frame #############
                #! for now, use the rightmost contour
                # for sprouts, use the downmost contour
                if len(xy_frame) > 1:
                    ind = np.argmax([xy[0] for xy in xy_frame])
                    # ind = np.argmax(radius_frame)
                    xy = xy_frame[ind]
                    pxy = pxy_frame[ind]
                elif len(xy_frame) == 1:
                    xy = xy_frame[0]
                    pxy = pxy_frame[0]
                # else:
                #     print('No contour detected!')
                if len(xy_init_traj) < num_init_frame and len(xy_frame) > 0:
                    xy_init_traj += [xy]
                    if len(xy_init_traj) == num_init_frame:
                        xy_init = np.mean(xy_init_traj, axis=0)
                        cnt_init = frame_cnt
                        pxy_traj += [pxy]

                # save trajectory
                if xy_init is not None and len(traj) == 0:
                    traj.append([xy])
                elif moving and abs(frame_cnt - target_mid_frame) <= 2:
                    mid_frame_candidate.append((xy, pxy, frame_cnt))
                elif moving and abs(frame_cnt - target_late_frame) <= 2:
                    late_frame_candidate.append((xy, pxy, frame_cnt))
                elif moving and abs(frame_cnt - target_end_frame) <= 2:
                    final_frame_candidate.append((xy, pxy, frame_cnt))

                # Check if the piece is moving, if so, calculate mid frame
                # if xy_init is not None and np.linalg.norm(xy - xy_init) > 0.02 and not moving:
                #     start_moving_frame = frame_cnt
                #     target_end_frame = frame_cnt + target_duration*fps
                #     target_mid_frame = int((target_end_frame - start_moving_frame)*mid_frame_ratio) + start_moving_frame
                #     print('Start moving at frame: ', start_moving_frame)
                #     moving = True

                # Update prev frame
                prev_frame = frame

            ##################################################################

            #! for now, if no contour detected for mid or final (e.g., piece under the spatula), we use the spatula tip position with some noise
            # Use the middle one in mid_frame_candidate
            if len(mid_frame_candidate) > 0:
                ind = int(len(mid_frame_candidate)/2)
                traj.append(mid_frame_candidate[ind][0])
                cnt_mid = mid_frame_candidate[ind][2]
                pxy_traj += [mid_frame_candidate[ind][1]]
            else:
                print('No mid frame candidate! Use spatula tip position instead.')
                tip_xy = [0.83+random.uniform(-0.01,0.01), 
                         random.uniform(-0.01,0.01)]
                traj.append(tip_xy)
                cnt_mid = target_mid_frame
                pxy_traj += [320, 288]

            # Use the middle one in late_frame_candidate
            if len(late_frame_candidate) > 0:
                ind = int(len(late_frame_candidate)/2)
                traj.append(late_frame_candidate[ind][0])
                cnt_late = late_frame_candidate[ind][2]
                pxy_traj += [late_frame_candidate[ind][1]]
            else:
                print('No late frame candidate! Use spatula tip position instead.')
                tip_xy = [0.86+random.uniform(-0.01,0.01), 
                         random.uniform(-0.01,0.01)]
                traj.append(tip_xy)
                cnt_late = target_mid_frame
                pxy_traj += [320, 288]

            # Use the middle one in final_frame_candidate
            if len(final_frame_candidate) > 0:
                ind = int(len(final_frame_candidate)/2)
                traj.append(final_frame_candidate[ind][0])
                cnt_end = final_frame_candidate[ind][2]
                pxy_traj += [final_frame_candidate[ind][1]]
            else:
                print('No final frame candidate! Use spatula tip position instead.')
                tip_xy = [0.90+random.uniform(-0.01,0.01), 
                         random.uniform(-0.01,0.01)]
                traj.append(tip_xy)
                cnt_end = target_end_frame
                pxy_traj += [320, 288]

            # close video
            cap.release()

            # sum traj
            traj_all.append(np.vstack(traj)[None])

            # Debug
            print('Trajectory:')
            print(np.array2string(traj_all[-1], separator=', '))
            print('Pixel XY of the detected contour:', pxy_traj)
            print('Frame number of the detected contour:', cnt_init, cnt_mid, cnt_end)
            fig, axes = plt.subplots(2, 4)
            axes[0,0].imshow(frame_traj[cnt_init])
            axes[0,1].imshow(frame_traj[cnt_mid])
            axes[0,2].imshow(frame_traj[cnt_late])
            axes[0,3].imshow(frame_traj[cnt_end])
            axes[1,0].imshow(hsv_traj[cnt_init])
            axes[1,1].imshow(hsv_traj[cnt_mid])
            axes[1,2].imshow(hsv_traj[cnt_late])
            axes[1,3].imshow(hsv_traj[cnt_end])
            axes[0,0].scatter(pxy_traj[0][0], pxy_traj[0][1], c='r', s=10)
            axes[0,1].scatter(pxy_traj[1][0], pxy_traj[1][1], c='r', s=10)
            axes[0,2].scatter(pxy_traj[2][0], pxy_traj[2][1], c='r', s=10)
            axes[0,3].scatter(pxy_traj[3][0], pxy_traj[3][1], c='r', s=10)
            plt.show()

        # Summary
        traj_all = np.vstack(traj_all)
        print('All states:')
        print(np.array2string(traj_all, separator=', '))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--num", required=True)
    args = vars(parser.parse_args())
    folder = args["folder"]
    num = args["num"]
    video_path_list = [os.path.join(folder, f'{ind}_k4a.avi') for ind in range(int(num))] 

    convert_env = ConvertEnv()
    convert_env.run(video_path_list)
