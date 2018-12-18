#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50  # Number of waypoints we will publish. You can change this number
FREQ_DELAY = .01
MAX_DECEL = 10


class WaypointUpdater(object):
    def __init__(self):
        self.base_waypoints = []
        self.waypoints_2d = None
        self.pose = None
        self.waypoints_tree = None

        self.waypoint_distance = None
        self.linear_velocity = 0
        self.stopline_wp_idx = -1

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        rospy.loginfo('{}: starting Waypoint Updater'.format(self.__class__.__name__))
        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.base_waypoints and self.pose and self.waypoints_tree:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        closest_waypoint_idx = self.waypoints_tree.query([x, y], 1)[1]

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

    def pose_cb(self, msg):
        # rospy.logwarn('got pose!')
        self.pose = msg

    def velocity_cb(self, msg):
        # rospy.logwarn(msg)
        self.linear_velocity = msg.twist.linear.x

    def waypoints_cb(self, waypoints):
        rospy.loginfo('{}: got base waypoints'.format(self.__class__.__name__))
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)


    def publish_waypoints(self):
        if self.waypoints_2d:
            final_lane = self.generate_lane()
            self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_waypoints.header
        closest_waypoint_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_waypoint_idx + LOOKAHEAD_WPS
        the_waypoints = self.base_waypoints.waypoints[closest_waypoint_idx: farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = the_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(the_waypoints, closest_waypoint_idx)

        return lane
        
    def decelerate_waypoints(self, waypoints, closest_idx):

        updated_waypoints = []
        stop_idx = max(self.stopline_wp_idx - closest_idx - 6, 6)
        
        for i, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            dist = self.distance(waypoints, i, stop_idx)

            vel = self.linear_velocity * (stop_idx - i - 1)/stop_idx
            if vel < 1.:
                vel = 0.

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            updated_waypoints.append(p)

        return updated_waypoints


    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
    
    def point_distance(self, point1, point2):
        x1 = point1[0]
        x2 = point2[0]
        y1 = point1[1]
        y2 = point2[1]

        return math.sqrt((x2-x1)**2 + (y2-y1)**2)



if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
