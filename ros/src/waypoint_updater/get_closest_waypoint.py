#!/usr/bin/env python

import rospy
# from waypoint_updater.srv import *
import os

def handle(req):
    pass


def init():
    print os.getcwd()
    rospy.init_node('get_closest_waypoint_service')
    rospy.logwarn('starting get closest waypoint service')
    # service = rospy.Service('get_closest_waypoint_service', GetClosestWaypoint, handle)

    rospy.spin()

if __name__ == '__main__':
    init()