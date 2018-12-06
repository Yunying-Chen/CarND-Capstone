#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from os import path, mkdir, getcwd
from cv_bridge import CvBridge
from io import BytesIO
import time
import cv2
import numpy as np

class StoplightImageCapturer:
    def __init__(self):
        rospy.init_node('stoplight_image_capture')

        rospy.Subscriber('/image_color', Image, self.image_cb)

        self.count = 0
        image_dir = rospy.get_param('image_dir')
        
        if not path.exists(image_dir):
            mkdir(image_dir)
        self.dir = path.join(image_dir, str(time.time()))
        mkdir(self.dir)

        self.bridge = CvBridge()

        rospy.spin()


    def image_cb(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        filename = path.join(self.dir, '{}.jpg'.format(self.count))
        cv2.imwrite(filename, image)
        self.count += 1

if __name__ == '__main__':
    StoplightImageCapturer()