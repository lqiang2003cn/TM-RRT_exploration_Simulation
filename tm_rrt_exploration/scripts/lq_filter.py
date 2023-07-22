#!/usr/bin/env python

import time
from copy import copy

# --------Include modules---------------
import rospy
import tf
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from numpy import array, vstack, delete, round
from numpy.linalg import norm
from sklearn.cluster import MeanShift
from std_msgs.msg import Bool
from tmrrt_exploration.msg import PointArray, invalidArray
from visualization_msgs.msg import Marker

import lq_utils as utils
from functions import information_gain


class Filter:

    def __init__(self, node_name, robot_name):
        self.map_data = OccupancyGrid()
        self.frontiers_points = None
        self.global_map = None
        self.local_map = None
        self.invalid_frontier = []
        self.start_signal = False
        self.reset_signal = False
        self.robot_name = robot_name
        self.threshold = None
        self.info_radius = None
        self.detected_points = None
        self.rate_hz = None
        self.global_costmap_topic = None
        self.local_map_topic = None
        self.bandwith_cluster = None
        self.robot_frame = None
        self.inv_frontier_topic = None
        self.local_map_sub = None
        self.compute_cycle = None
        self.start_signal_topic = None
        self.reset_signal_topic = None
        self.rate = None
        self.map_topic = None
        self.tf_listener = None
        self.global_frame = None
        self.frontier_points_pub = None
        self.centroid_points_pub = None
        self.filtered_points_pub = None
        self.points = None
        self.node_name = node_name
        self.points = None
        self.centroids_points = None

    def init(self):
        self.map_topic = rospy.get_param('~map_topic', '/map')
        self.threshold = rospy.get_param('~costmap_clearing_threshold', 70)
        self.info_radius = rospy.get_param('~info_radius', 1.0)
        self.detected_points = rospy.get_param('~detected_points', '/detected_points')
        self.rate_hz = rospy.get_param('~rate', 100)
        self.global_costmap_topic = rospy.get_param('~global_costmap_topic', '/move_base_node/global_costmap/costmap')
        self.local_map_topic = rospy.get_param('~local_map', self.robot_name + '/map')
        self.bandwith_cluster = rospy.get_param('~bandwith_cluster', 0.3)
        self.robot_frame = rospy.get_param('~robot_frame', self.robot_name + '/base_footprint')
        self.inv_frontier_topic = rospy.get_param('~invalid_frontier', '/invalid_points')
        self.compute_cycle = rospy.get_param('~compute_cycle', 0.0)
        self.start_signal_topic = rospy.get_param('~start_signal_topic', '/explore_start')
        self.reset_signal_topic = rospy.get_param('~reset_signal_topic', '/explore_reset')
        self.rate = rospy.Rate(self.rate_hz)

        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_call_back)
        rospy.Subscriber(self.inv_frontier_topic, invalidArray, self.invalid_call_back)
        rospy.Subscriber(self.start_signal_topic, Bool, self.start_signal_call_back)
        rospy.Subscriber(self.reset_signal_topic, Bool, self.reset_signal_topic)

        # ---------------------------------------------------------------------------------------------------------------
        self.global_map = OccupancyGrid()
        self.local_map = OccupancyGrid()

        rospy.Subscriber(self.robot_name + self.global_costmap_topic, OccupancyGrid, self.global_cost_map_call_back)
        if self.local_map_sub:
            rospy.Subscriber(self.robot_name + self.local_map_topic, OccupancyGrid, self.local_map_call_back)

        while len(self.map_data.data) < 1:
            rospy.loginfo('Waiting for the map')
            rospy.sleep(0.1)
            pass

        while len(self.global_map.data) < 1:
            rospy.loginfo('Waiting for the global costmap')
            rospy.sleep(0.1)
            pass

        if self.local_map_sub:
            while len(self.local_map.data.data) < 1:
                rospy.loginfo('Waiting for the local map')
                rospy.sleep(0.1)
                pass

        global_frame = "/" + self.map_data.header.frame_id

        try:
            self.tf_listener = tf.TransformListener()
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.sleep(0.1)
            pass

        rospy.loginfo('Waiting for TF Transformer')
        self.tf_listener.waitForTransform(global_frame[1:], self.robot_frame, rospy.Time(0), rospy.Duration(10))

        rospy.Subscriber(
            self.detected_points,
            PointStamped,
            callback=self.frontier_call_back,
            callback_args=[self.tf_listener, self.global_frame[1:]]
        )

        self.frontier_points_pub = rospy.Publisher('/frontiers', Marker, queue_size=10)
        self.centroid_points_pub = rospy.Publisher('/centroids', Marker, queue_size=10)
        self.filtered_points_pub = rospy.Publisher('/filtered_points', PointArray, queue_size=10)

        rospy.loginfo("the map and global costmaps are received")

        while len(self.frontiers_points) < 1:
            pass

        self.points = Marker()
        self.points.header.frame_id = self.map_data.header.frame_id
        self.points.header.stamp = rospy.Time.now()
        self.points.ns = "frontiers"
        self.points.id = 0
        self.points.type = Marker.POINTS
        self.points.action = Marker.ADD
        self.points.pose.orientation.w = 1.0
        self.points.scale.x = 0.2
        self.points.scale.y = 0.2
        self.points.color.r = 255.0 / 255.0
        self.points.color.g = 255.0 / 255.0
        self.points.color.b = 0.0 / 255.0
        self.points.color.a = 0.5
        self.points.lifetime = rospy.Duration()

        self.centroids_points = Marker()
        self.centroids_points.header.frame_id = self.map_data.header.frame_id
        self.centroids_points.header.stamp = rospy.Time.now()
        self.centroids_points.ns = "centroid"
        self.centroids_points.id = 4
        self.centroids_points.type = Marker.POINTS
        self.centroids_points.action = Marker.ADD
        self.centroids_points.pose.orientation.w = 1.0
        self.centroids_points.scale.x = 0.4
        self.centroids_points.scale.y = 0.4
        self.centroids_points.color.r = 0.0 / 255.0
        self.centroids_points.color.g = 255.0 / 255.0
        self.centroids_points.color.b = 0.0 / 255.0
        self.centroids_points.color.a = 1
        self.centroids_points.lifetime = rospy.Duration()

    def reset(self):
        self.frontiers_points = []
        self.invalid_frontier = []

    def do_filtering(self):
        temp_point = PointStamped()
        temp_point.header.frame_id = self.map_data.header.frame_id
        temp_point.header.stamp = rospy.Time(0)
        temp_point.point.z = 0.0
        filtered_points = PointArray()
        # temp_point = Point()
        # temp_point.z = 0.0
        next_compute_time = time.time()
        if time.time() + self.compute_cycle >= next_compute_time:  # why
            centroids = []
            front = copy(self.frontiers_points)
            if len(front) > 1:
                ms = MeanShift(bandwidth=self.bandwith_cluster)
                ms.fit(front)
                centroids = ms.cluster_centers_  # centroids array is the centers of each cluster
            elif len(front) == 1:
                centroids = front

            # self.frontiers = copy(centroids)

            z = 0
            while z < len(centroids):
                cond1 = False
                cond3 = False
                temp_point.point.x = centroids[z][0]
                temp_point.point.y = centroids[z][1]

                trans_point = self.tf_listener.transformPoint(self.global_map.header.frame_id, temp_point)
                trans_pts_np = array([trans_point.point.x, trans_point.point.y])
                cond1 = (utils.grid_value(self.global_map, trans_pts_np) > self.threshold) or cond1

                for inv_frt in range(0, len(self.invalid_frontier)):
                    if norm(centroids[z], self.invalid_frontier[inv_frt]) < 0.1:
                        cond2 = True

                map_value = utils.grid_value_merged_map(self.map_data, centroids[z])
                if map_value > 90:  # if the map value is unknown or obstacle
                    cond3 = True
                info_gain = information_gain(self.map_data, centroids[z], self.info_radius * 0.5)

                if cond1 or cond3 or info_gain < 0.2:
                    centroids = delete(centroids, z, axis=0)
                    z = z - 1
                z += 1

            # publishing
            filtered_points.points = []
            for i in range(0, len(centroids)):
                invalid_pts = False
                for j in range(0, len(self.invalid_frontier)):
                    if self.invalid_frontier[j][0] == centroids[i][0] and self.invalid_frontier[j][1] == centroids[i][1]:
                        invalid_pts = True
                if not invalid_pts:
                    tp = Point()
                    tp.x = round(centroids[i][0], 2)
                    tp.y = round(centroids[i][1], 2)
                    tp.z = 0.0
                    filtered_points.points.append(temp_point)

            pp = []
            for q in range(0, len(self.frontiers_points)):
                p = Point()
                p.x = self.frontiers_points[q][0]
                p.y = self.frontiers_points[q][1]
                pp.append(copy(p))
            self.frontiers_points.points = pp

            pp = []
            for q in range(0, len(centroids)):
                p = Point()
                p.x = centroids[q][0]
                p.y = centroids[q][1]
                pp.append(p)
            self.centroids_points.points = pp

            self.filtered_points_pub.publish(filtered_points)
            self.frontier_points_pub.publish(self.frontiers_points)
            self.centroid_points_pub.publish(self.centroids_points)

        self.rate.sleep()

    def loop(self):
        while not rospy.is_shutdown():
            if self.reset_signal:
                self.reset()
                self.rate.sleep()
            else:
                if self.start_signal:
                    self.do_filtering()
                else:
                    self.rate.sleep()

    def frontier_call_back(self, data, args):
        transformed_point = args[0].transformPoint(args[1], data)
        x = [array([transformed_point.point.x, transformed_point.point.y])]

        if len(self.frontiers_points) > 0:
            self.frontiers_points = vstack((self.frontiers_points, x))
        else:
            self.frontiers_points = x

    def map_call_back(self, data):
        self.map_data = data

    def invalid_call_back(self, data):
        self.invalid_frontier = []
        for point in data.points:
            self.invalid_frontier.append(array([point.x, point.y]))

    def local_map_call_back(self, data):
        self.local_map = data

    def global_cost_map_call_back(self, data):
        self.global_map = data

    def start_signal_call_back(self, data):
        self.start_signal = data.data

    def reset_signal_call_back(self, data):
        self.reset_signal = data.data


if __name__ == '__main__':
    try:
        robot_name = 'tb3_0'
        func_node_name = 'filter'
        node_name = func_node_name + '_' + robot_name
        rospy.init_node(node_name, anonymous=False)

        frontier_filter = Filter(node_name, robot_name)
        frontier_filter.init()
        frontier_filter.loop()
    except rospy.ROSInterruptException:
        pass
