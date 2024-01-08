#!/usr/bin/env python3
# https://stackoverflow.com/questions/72661821/closing-a-videowriter-in-a-ros2-node
import rospy
from sensor_msgs.msg import Image
import cv2
import cv_bridge
from franka_irom.srv import TriggerRecord, TriggerRecordResponse 
import rosbag


class RsRecordEnv(object):
    """Recording video with realsense, with ROS service for starting and finishing recording."""

    def __init__(self):
        super(RsRecordEnv, self).__init__()

        self.flag_recording = False
        # self.fps = 30
        self.bag = None
        # self.video_size = (1920, 1080)
        # self.flag_trigger = False

        # Used to convert between ROS and OpenCV images
        self.bridge = cv_bridge.CvBridge()

        # Node is subscribing to the video_frames topic
        rospy.Subscriber('/rs_camera/color/image_raw', Image, self.__img_callback)

        # Initialize service
        self.pos_pub = rospy.Service('~trigger_record', TriggerRecord, self.__trigger_record)
        # bool success, string message - use string as file name


    def __img_callback(self, msg):
        if self.flag_recording and self.bag is not None:
            self.bag.write('image', msg, msg.header.stamp)
            # np_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")  # 1920, 1080, 4
            # self.vid_writer.write(np_img[:, :, :3])


    def __trigger_record(self, req):
        res = TriggerRecordResponse()
        # self.flag_trigger = True
        self.file_name = req.file_name

        if self.flag_recording:
            self.flag_recording = False
            self.bag.close()
            self.bag = None
            # self.vid_writer.release()
            # self.vid_writer = None
            rospy.loginfo("Finish recording rs!")
        else:
            self.flag_recording = True
            self.file_name = req.file_name

            self.bag = rosbag.Bag(self.file_name, 'w')
            # self.vid_writer = cv2.VideoWriter(
            #     filename=self.file_name,
            #     apiPreference=cv2.CAP_FFMPEG,
            #     fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
            #     fps=self.fps,
            #     frameSize=self.video_size,
            # )
            rospy.loginfo(f"Start recording rs with name {req.file_name}!")
        res.success = True
        res.message = "done"
        return res


if __name__ == '__main__':
    rospy.init_node('rs_record_env')
    rs_record_env = RsRecordEnv()
    rospy.spin()
