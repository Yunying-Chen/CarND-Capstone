#!/usr/bin/env python

import rospy
from waypoint_updater.srv import GetClosestWaypoint, GetClosestWaypointResponse
from styx_msgs.msg import Lane
import math
import numpy as np
from scipy.spatial import KDTree


class GetClosestWaypointService:
    def __init__(self):
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.linear_velocity = None
        self.waypoint_distance = None

        rospy.init_node('get_closest_waypoint_service')

        rospy.Subscriber('/base_waypoints', Lane, self.base_waypoints_cb)
        # rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        service = rospy.Service('get_closest_waypoint_service', GetClosestWaypoint, self.handle)

        print 'running: {}'.format(GetClosestWaypointService.__name__)
        service.spin()

    def handle(self, req):
        print 'x: {}'.format(req.x)
        print 'y: {}'.format(req.y)
        return GetClosestWaypointResponse(self.get_closest_waypoint_idx(req.x, req.y))

    def base_waypoints_cb(self, msg):
        self.base_waypoints = msg
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 self.base_waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)
            self.waypoint_distance = self.point_distance(self.waypoints_2d[0], self.waypoints_2d[1])

    def velocity_cb(self, msg):
        # rospy.logwarn(msg)
        self.linear_velocity = msg.twist.linear.x

    def point_distance(self, point1, point2):
        x1 = point1[0]
        x2 = point2[0]
        y1 = point1[1]
        y2 = point2[1]

        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def get_closest_waypoint_idx(self, x, y):
        closest_waypoint_idx = self.waypoints_tree.query([x, y], 1)[1]

        # num_of_waypoints_to_avance = int(math.floor(self.linear_velocity * FREQ_DELAY / self.waypoint_distance))

        # closest_waypoint_idx += num_of_waypoints_to_avance

        closest_coord = self.waypoints_2d[closest_waypoint_idx]
        prev_coord = self.waypoints_2d[closest_waypoint_idx - 1]

        # Check if point is in front
        cl_vec = np.array(closest_coord)
        prev_vec = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vec - prev_vec, pos_vect - cl_vec)

        if val > 0:
            closest_waypoint_idx = (closest_waypoint_idx + 1) % len(self.waypoints_2d)
        return closest_waypoint_idx


if __name__ == '__main__':
    GetClosestWaypointService()
