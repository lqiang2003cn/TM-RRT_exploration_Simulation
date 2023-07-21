#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import Point, PolygonStamped, Polygon, PointStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Bool


#############################################################
class Boundary:

    def __init__(self, node_name):
        self.boundary_points = []
        self.recorded_points = []
        self.start_exploration = True
        self.reset_exploration = False
        self.valid_entry = False
        self.odom_points = [0.0, 0.0]
        self.node_name = node_name

        self.data_length = None
        self.map_frame = None
        self.num_points = None
        self.clicked_point = None
        self.topic_output = None
        self.start_topic = None
        self.reset_topic = None
        self.control_output = None
        self.reset_control = None
        self.frequency = None
        self.time_interval = None
        self.auto_input_point = None
        self.diag_distance = None
        self.map_topic = None
        self.initial_point = None
        self.odom_topic = None
        self.robot_frame = None
        self.start_control_pub = None
        self.reset_control_pub = None
        self.clicked_point_pub = None
        self.rate = None
        self.exploration_boundary = None
        self.boundary_polygon_pub = None

    def init(self):
        robot_name = '/tb3_0'
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.num_points = rospy.get_param('~n_point', 4)
        self.clicked_point = rospy.get_param('~clicked_point', '/clicked_point')
        self.exploration_boundary = rospy.get_param('~exploration_boundary', '/exploration_boundary')
        self.start_topic = rospy.get_param('~start_topic', '/start_signal')
        self.reset_topic = rospy.get_param('~reset_topic', '/reset_signal')
        self.control_output = rospy.get_param('~control_output', '/explore_start')
        self.reset_control = rospy.get_param('~restart_output', '/explore_reset')
        self.frequency = rospy.get_param('~frequency', 2.0)
        self.auto_input_point = rospy.get_param('~auto_input_point', "")
        self.diag_distance = rospy.get_param('~diagonal_distance', 7.5)
        self.initial_point = rospy.get_param('~initial_point', '-1')
        self.map_topic = rospy.get_param('~map_topic', robot_name + '/map')
        self.odom_topic = rospy.get_param('~odom_topic', robot_name + '/odom')
        self.robot_frame = rospy.get_param('~robot_frame', robot_name + '/base_footprint')

        rospy.Subscriber(self.start_topic, Bool, self.start_exploration_callback)
        rospy.Subscriber(self.reset_topic, Bool, self.reset_exploration_callback)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.boundary_map_info_callback)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)

        self.start_control_pub = rospy.Publisher(self.control_output, Bool, queue_size=1)
        self.reset_control_pub = rospy.Publisher(self.reset_control, Bool, queue_size=1)
        self.clicked_point_pub = rospy.Publisher(self.clicked_point, PointStamped, queue_size=10)
        self.boundary_polygon_pub = rospy.Publisher(self.exploration_boundary, PolygonStamped, queue_size=10)

        self.rate = rospy.Rate(self.frequency)
        self.data_length = self.num_points

    def get_boundary_points_msg(self):
        res = []
        for bp in self.boundary_points:
            temp_points = Point()
            temp_points.x = bp[0]
            temp_points.y = bp[1]
            temp_points.z = 0
            res.append(temp_points)
        return res

    def start_the_exploration(self):
        if not self.valid_entry:
            while len(self.boundary_points) < 4:
                print(self.boundary_points)
                print('waiting for the boundary points to be calculated')
                self.rate.sleep()

            for point in self.boundary_points:
                pub_point1 = PointStamped()
                pub_point1.header.frame_id = self.map_frame
                pub_point1.point.x = float(point[0])
                pub_point1.point.y = float(point[1])
                pub_point1.point.z = 0.0
                self.clicked_point_pub.publish(pub_point1)

            pub_point2 = PointStamped()
            pub_point2.header.frame_id = self.map_frame
            if self.initial_point == '-1':
                pub_point2.point.x = float(self.odom_points[0])
                pub_point2.point.y = float(self.odom_points[1])
            else:
                temp_str = self.initial_point.split(",")
                pub_point2.point.x = float(temp_str[0])
                pub_point2.point.y = float(temp_str[1])
            pub_point2.point.z = 0.0
            self.clicked_point_pub.publish(pub_point2)
            rospy.sleep(1)
            self.valid_entry = True

        if self.valid_entry:
            # check only 4 points input by the user.
            start_time = rospy.get_rostime().secs
            rospy.loginfo('--- >>> the exploration starts at time: %.2f ' % start_time)
            while self.start_exploration:
                header = Header()
                boundary_list = PolygonStamped()
                boundary_points = Polygon()
                boundary_points.points = self.get_boundary_points_msg()
                boundary_list.polygon = boundary_points
                header.frame_id = self.map_frame
                header.stamp = rospy.Time.now()
                boundary_list.header = header
                self.boundary_polygon_pub.publish(boundary_list)

                control_msg = Bool()
                control_msg.data = self.start_exploration
                self.start_control_pub.publish(control_msg)
                reset_msg = Bool()
                reset_msg.data = self.reset_exploration
                self.reset_control_pub.publish(reset_msg)
                self.rate.sleep()

    def stop_the_exploration(self):
        control_msg = Bool()
        control_msg.data = self.start_exploration
        self.start_control_pub.publish(control_msg)
        reset_msg = Bool()
        reset_msg.data = False
        self.reset_control_pub.publish(reset_msg)
        self.rate.sleep()

    def reset_the_exploration(self):
        control_msg = Bool()
        control_msg.data = self.start_exploration
        self.start_control_pub.publish(control_msg)
        reset_msg = Bool()
        reset_msg.data = self.reset_exploration
        self.reset_control_pub.publish(reset_msg)

        reset_msg.data = False
        self.reset_control_pub.publish(reset_msg)
        self.reset_exploration = False
        if len(self.recorded_points) > 0:
            self.recorded_points = []
            self.valid_entry = False

    def loop(self):
        if not self.reset_exploration:
            if self.start_exploration:
                self.start_the_exploration()
            else:
                self.stop_the_exploration()
        else:
            self.reset_the_exploration()
        self.rate.sleep()

    def boundary_map_info_callback(self, data):
        map_resolution = data.info.resolution
        map_width = data.info.height
        map_height = data.info.width
        pos_origin_x = data.info.origin.position.x
        pos_origin_y = data.info.origin.position.y

        if len(self.boundary_points) < 4:
            self.boundary_points.append(np.array([((map_width * map_resolution) * -1.0) - pos_origin_x + 0.01,
                                                  ((map_height * map_resolution) * 1.0) + pos_origin_y - 0.01]))
            self.boundary_points.append(np.array([((map_width * map_resolution) * -1.0) - pos_origin_x + 0.01,
                                                  ((map_height * map_resolution) * -1.0) - pos_origin_y + 0.01]))
            self.boundary_points.append(np.array([((map_width * map_resolution) * 1.0) + pos_origin_x - 0.01,
                                                  ((map_height * map_resolution) * -1.0) - pos_origin_y + 0.01]))
            self.boundary_points.append(np.array([((map_width * map_resolution) * 1.0) + pos_origin_x - 0.01,
                                                  ((map_height * map_resolution) * 1.0) + pos_origin_y - 0.01]))

    def start_exploration_callback(self, data):
        if data.data and not self.start_exploration:
            self.start_exploration = True
        elif not data.data and self.start_exploration:
            self.start_exploration = False
        if data.data and self.start_exploration:
            self.start_exploration = True

    def reset_exploration_callback(self, data):
        if data.data:
            self.start_exploration = False
            self.reset_exploration = True

    #############################################################
    def odom_callback(self, data):
        self.odom_points = [data.pose.pose.position.x, data.pose.pose.position.y]


if __name__ == '__main__':
    try:
        nn = 'exploration_boundary'
        rospy.init_node(nn, anonymous=False)
        boundary = Boundary(nn)
        boundary.init()
        boundary.loop()
    except rospy.ROSInterruptException:
        pass
