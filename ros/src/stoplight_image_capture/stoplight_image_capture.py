#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from os import path, mkdir, getcwd
from cv_bridge import CvBridge
from io import BytesIO
import time
import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
import json

from styx_msgs.msg import TrafficLightArray, TrafficLight

from waypoint_updater.srv import GetClosestWaypoint, GetClosestWaypointResponse


class StoplightImageCapturer:
    def __init__(self):
        self.lights = None
        self.pose = None

        rospy.init_node('stoplight_image_capture')
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        rospy.Subscriber('/image_color', Image, self.image_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

        self.get_closest_waypoint_service = rospy.ServiceProxy('/get_closest_waypoint_service', GetClosestWaypoint)

        self.count = 0
        image_dir = rospy.get_param('image_dir')
        
        if not path.exists(image_dir):
            mkdir(image_dir)
        self.dir = path.join(image_dir, str(time.time()))
        mkdir(self.dir)

        self.bridge = CvBridge()

        rospy.spin()

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def pose_cb(self, msg):
        self.pose = msg

    def image_cb(self, msg):
        closest_light_waypoint_idx = None
        closest_light = None

        # for i, light in enumerate(self.lights):
        #     light_coord = stop_line_positions[i]
        #     test_light_waypoint_idx = self.get_closest_waypoint(light_coord[0], light_coord[1])
        #     d = test_light_waypoint_idx - current_waypoint_idx
        #     if 0 < d < diff:
        #         closest_light_waypoint_idx = test_light_waypoint_idx
        #         closest_light = light
        #         diff = d

        if self.lights:
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y
            closest_car_point_idx = self.get_closest_waypoint_service(x, y).current_waypoint_idx
            rospy.logwarn('car is at {}'.format(closest_car_point_idx))
            closest_light = None
            closest_dist = 1000000
            closest_index = None
            for l in self.lights:
                lx = l.pose.pose.position.x
                ly = l.pose.pose.position.y
                idx = self.get_closest_waypoint_service(lx, ly).current_waypoint_idx
                dist = idx - closest_car_point_idx
                if 0 < dist < closest_dist:
                    closest_light = l
                    closest_index = idx
                    closest_dist = dist

            data = {
                'carWp': closest_car_point_idx,
                'lightWp': closest_index,
                'lightState': closest_light.state
            }

            data_filename = path.join(self.dir, '{}.json'.format(self.count))
            with open(data_filename, 'w') as f:
                f.write(json.dumps(data))

            image = self.bridge.imgmsg_to_cv2(msg)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            filename = path.join(self.dir, '{}.jpg'.format(self.count))
            cv2.imwrite(filename, image)
            self.count += 1


if __name__ == '__main__':
    StoplightImageCapturer()