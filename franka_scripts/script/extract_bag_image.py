import argparse
import os
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
import rosbag
# from sensor_msgs.msg import Image
import glob

cv_bridge = CvBridge()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    args = vars(parser.parse_args())
    folder = args["folder"]
    bag_path_list = glob.glob(os.path.join(folder, '*rs.bag'))

    # parameters
    fps = 30
    video_size = (1920, 1080)

    # Process each file
    for i, bag_path in enumerate(bag_path_list):
        print(f'Processing {bag_path}...')

        video_path = bag_path.split('.')[0] + '.mp4'
        vid_writer = cv2.VideoWriter(
                filename=video_path,
                apiPreference=cv2.CAP_FFMPEG,
                fourcc=cv2.VideoWriter_fourcc(*"avc1"),
                fps=fps,
                frameSize=video_size,
            )
        
        # try:
        bag = rosbag.Bag(bag_path)
        # except:
            # print('Cannot open bag file!')
            # continue

        for topic, msg, t in bag.read_messages(topics=['image']):
            cv_img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            vid_writer.write(cv_img[:, :, :3])
        vid_writer.release()
        bag.close()
