#!/usr/bin/env python

from copy import copy

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


class Filter:

    def __init__(self, robot_name):
        self.map_data = OccupancyGrid()
        self.frontiers_points = None
        self.global_map = None
        self.local_map = None
        self.invalid_frontiers = []
        self.start_signal = True
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
        self.invalid_frontiers_topic = None
        self.start_signal_topic = None
        self.reset_signal_topic = None
        self.rate = None
        self.map_topic = None
        self.tf_listener = None
        self.global_frame = None
        self.frontier_points_pub = None
        self.filtered_points_shape_pub = None
        self.filtered_points_pub = None
        self.frontiers_points = None
        self.frontiers_points = None
        self.filtered_points_shape = None
        self.global_frame = None
        self.frontiers = None

    def init(self):
        self.map_topic = rospy.get_param('~map_topic', self.robot_name + '/map')
        self.threshold = rospy.get_param('~costmap_clearing_threshold', 70)
        self.info_radius = rospy.get_param('~info_radius', 1.0)
        self.detected_points = rospy.get_param('~detected_points', '/detected_points')
        self.rate_hz = rospy.get_param('~rate', 100)
        self.global_costmap_topic = rospy.get_param('~global_costmap_topic', '/move_base_node/global_costmap/costmap')
        self.local_map_topic = rospy.get_param('~local_map', '/map')
        self.bandwith_cluster = rospy.get_param('~bandwith_cluster', 0.3)
        self.robot_frame = rospy.get_param('~robot_frame', self.robot_name + '/base_footprint')
        self.invalid_frontiers_topic = rospy.get_param('~invalid_frontiers', '/invalid_frontiers')
        self.start_signal_topic = rospy.get_param('~start_signal_topic', '/explore_start')
        self.reset_signal_topic = rospy.get_param('~reset_signal_topic', '/explore_reset')
        self.reset_signal_topic = rospy.get_param('~reset_signal_topic', '/explore_reset')
        self.rate = rospy.Rate(self.rate_hz)
        self.global_map = OccupancyGrid()
        self.frontiers = []

        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_call_back)
        rospy.Subscriber(self.invalid_frontiers_topic, invalidArray, self.invalid_call_back)
        rospy.Subscriber(self.start_signal_topic, Bool, self.start_signal_call_back)
        rospy.Subscriber(self.reset_signal_topic, Bool, self.reset_signal_topic)
        rospy.Subscriber(self.robot_name + self.global_costmap_topic, OccupancyGrid, self.global_cost_map_call_back)

        # self.frontier_points_pub = rospy.Publisher('/frontiers', Marker, queue_size=10)
        self.filtered_points_shape_pub = rospy.Publisher('/filtered_points_shapes', Marker, queue_size=10)
        self.filtered_points_pub = rospy.Publisher('/filtered_points', PointArray, queue_size=10)

        while len(self.map_data.data) < 1:
            rospy.loginfo('Waiting for the map')
            rospy.sleep(0.1)
            pass

        while len(self.global_map.data) < 1:
            rospy.loginfo('Waiting for the global costmap')
            rospy.sleep(0.1)
            pass
        self.global_frame = self.map_data.header.frame_id

        try:
            self.tf_listener = tf.TransformListener()
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.sleep(0.1)
            pass

        rospy.loginfo('Waiting for TF Transformer')
        self.tf_listener.waitForTransform(self.global_frame, self.robot_frame, rospy.Time(0), rospy.Duration(10))

        rospy.Subscriber(
            self.detected_points, PointStamped, callback=self.frontier_call_back,
            callback_args=[self.tf_listener, self.global_frame]
        )

        self.init_markers()
        rospy.loginfo("the map and global costmaps are received")

        while len(self.frontiers) < 1:
            pass
        rospy.loginfo("frontiers are received")

    def init_markers(self):
        self.frontiers_points = Marker()
        self.frontiers_points.header.frame_id = self.map_data.header.frame_id
        self.frontiers_points.header.stamp = rospy.Time.now()
        self.frontiers_points.ns = "frontiers"
        self.frontiers_points.id = 0
        self.frontiers_points.type = Marker.POINTS
        self.frontiers_points.action = Marker.ADD
        self.frontiers_points.pose.orientation.w = 1.0
        self.frontiers_points.scale.x = 0.2
        self.frontiers_points.scale.y = 0.2
        self.frontiers_points.color.r = 255.0 / 255.0
        self.frontiers_points.color.g = 255.0 / 255.0
        self.frontiers_points.color.b = 0.0 / 255.0
        self.frontiers_points.color.a = 0.5
        self.frontiers_points.lifetime = rospy.Duration()

        self.filtered_points_shape = Marker()
        self.filtered_points_shape.header.frame_id = self.map_data.header.frame_id
        self.filtered_points_shape.header.stamp = rospy.Time.now()
        self.filtered_points_shape.ns = "centroid"
        self.filtered_points_shape.id = 4
        self.filtered_points_shape.type = Marker.POINTS
        self.filtered_points_shape.action = Marker.ADD
        self.filtered_points_shape.pose.orientation.w = 1.0
        self.filtered_points_shape.scale.x = 0.4
        self.filtered_points_shape.scale.y = 0.4
        self.filtered_points_shape.color.r = 0.0 / 255.0
        self.filtered_points_shape.color.g = 255.0 / 255.0
        self.filtered_points_shape.color.b = 0.0 / 255.0
        self.filtered_points_shape.color.a = 1
        self.filtered_points_shape.lifetime = rospy.Duration()

    def reset(self):
        self.frontiers_points = []
        self.invalid_frontiers = []

    def do_filtering(self):
        temp_point = PointStamped()
        temp_point.header.frame_id = self.map_data.header.frame_id
        temp_point.header.stamp = rospy.Time(0)
        temp_point.point.z = 0.0
        filtered_points = PointArray()

        centroids = []
        frontiers_snapshot = copy(self.frontiers)

        if len(frontiers_snapshot) > 1:
            ms = MeanShift(bandwidth=self.bandwith_cluster)
            ms.fit(frontiers_snapshot)
            centroids = ms.cluster_centers_  # centroids array is the centers of each cluster
        elif len(frontiers_snapshot) == 1:
            centroids = frontiers_snapshot

        # make frontiers smaller but not empty
        self.frontiers = copy(centroids)

        z = 0
        while z < len(centroids):
            invalid_cnd = False
            map_cnd = False
            temp_point.point.x = centroids[z][0]
            temp_point.point.y = centroids[z][1]
            trans_point = self.tf_listener.transformPoint(self.global_map.header.frame_id, temp_point)
            trans_pts_np = array([trans_point.point.x, trans_point.point.y])

            # global map check
            global_cnd = (utils.grid_value(self.global_map, trans_pts_np) > self.threshold)

            # invalid point check
            for inv_frt in range(0, len(self.invalid_frontiers)):
                if norm(centroids[z], self.invalid_frontiers[inv_frt]) < 0.1:
                    invalid_cnd = True

            map_value = utils.grid_value_merged_map(self.map_data, centroids[z])
            if map_value > 90:  # if the map value is unknown or obstacle
                map_cnd = True

            info_gain = utils.information_gain(self.map_data, centroids[z], self.info_radius * 0.5)

            if global_cnd or invalid_cnd or map_cnd or info_gain < 0.2:
                centroids = delete(centroids, z, axis=0)
                z = z - 1
            z += 1

        # publishing
        filtered_points.points = []
        for i in range(0, len(centroids)):
            invalid_pts = False
            for j in range(0, len(self.invalid_frontiers)):
                if self.invalid_frontiers[j][0] == centroids[i][0] and self.invalid_frontiers[j][1] == centroids[i][1]:
                    invalid_pts = True
            if not invalid_pts:
                tp = Point()
                tp.x = round(centroids[i][0], 2)
                tp.y = round(centroids[i][1], 2)
                tp.z = 0.0
                filtered_points.points.append(tp)

        if len(filtered_points.points) > 0:
            # pp = []
            # for q in range(0, len(frontiers_snapshot)):
            #     p = Point()
            #     p.x = frontiers_snapshot[q][0]
            #     p.y = frontiers_snapshot[q][1]
            #     p.z = 0
            #     pp.append(p)
            # self.frontiers_points.points = pp

            pp = []
            for q in range(0, len(filtered_points.points)):
                p = Point()
                p.x = filtered_points.points[q].x
                p.y = filtered_points.points[q].y
                p.z = filtered_points.points[q].z
                pp.append(p)
            self.filtered_points_shape.points = pp

            # the same filtered points pass to different destinations
            self.filtered_points_pub.publish(filtered_points)
            self.filtered_points_shape_pub.publish(self.filtered_points_shape)
            # self.frontier_points_pub.publish(self.frontiers_points)

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

        if len(self.frontiers) > 0:
            self.frontiers = vstack((self.frontiers, x))
        else:
            self.frontiers = x

    def map_call_back(self, data):
        self.map_data = data

    def invalid_call_back(self, data):
        self.invalid_frontiers = []
        for point in data.points:
            self.invalid_frontiers.append(array([point.x, point.y]))

    def local_map_call_back(self, data):
        self.local_map = data

    def global_cost_map_call_back(self, data):
        self.global_map = data

    def start_signal_call_back(self, data):
        self.start_signal = data.data

    def reset_signal_call_back(self, data):
        self.reset_signal = data.data
