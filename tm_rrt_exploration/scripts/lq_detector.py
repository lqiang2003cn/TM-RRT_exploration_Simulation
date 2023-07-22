import numpy as np
import rospy
import tf
from geometry_msgs.msg import PointStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from numpy.linalg import norm
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker

import lq_utils as utils


class Detector:

    def __init__(self, node_name, mode, robot_name):
        self.start_x = None
        self.start_y = None
        self.init_map_x = None
        self.init_map_y = None
        self.start_signal = True
        self.reset_signal = False
        self.occupy_map_data = OccupancyGrid()
        self.boundary_points = Marker()
        self.line = Marker()
        self.points = Marker()
        self.first_run = True
        self.V = []
        self.eta = 1
        self.map_topic = None
        self.rate_hz = 100
        self.detected_points_pub = None
        self.shapes_pub = None
        self.node_name = node_name
        self.rate = None
        self.mode = mode
        self.odom_points = None
        self.odom_topic = None
        self.tf_listener = tf.TransformListener()
        self.map_frame = None
        self.robot_frame = None
        self.robot_name = robot_name

    def init(self):
        self.eta = rospy.get_param('~eta', 0.5)
        self.map_topic = rospy.get_param('~map_topic', "/map")
        self.map_frame = rospy.get_param('~map_frame', "map")
        self.rate_hz = rospy.get_param('~rate', 100)
        self.odom_topic = rospy.get_param('~odom_topic', self.robot_name + '/odom')
        self.robot_frame = rospy.get_param('~robot_frame', self.robot_name + '/base_footprint')

        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback, queue_size=100)
        rospy.Subscriber("/explore_start", Bool, self.start_signal_callback, queue_size=10)
        rospy.Subscriber("/explore_reset", Bool, self.reset_signal_callBack, queue_size=10)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)

        self.detected_points_pub = rospy.Publisher('/detected_points', PointStamped, queue_size=100)
        self.shapes_pub = rospy.Publisher(self.node_name + '_shapes', Marker, queue_size=100)

        self.rate = rospy.Rate(self.rate_hz)

        while len(self.occupy_map_data.data) < 1:
            rospy.loginfo('Waiting for the map')
            self.rate.sleep()
            pass

        while self.odom_points is None:
            rospy.loginfo('Waiting for odom')
            self.rate.sleep()
            pass

        rospy.loginfo('map received')
        rospy.loginfo('odom received')

    def loop(self):
        while not rospy.is_shutdown():
            if self.reset_signal:
                self.reset()
            else:
                if self.start_signal:
                    if self.first_run:
                        self.handle_first_run()

                    if not self.first_run:
                        self.detect()
                else:
                    self.rate.sleep()

    def handle_first_run(self):
        self.init_visual_info()
        self.setup_boundary_points()
        self.cal_init_info()
        self.first_run = False

    def cal_init_info(self):
        temp1 = np.array([self.boundary_points.points[0][0], self.boundary_points.points[0][1]])
        temp2 = np.array([self.boundary_points.points[2][0], self.boundary_points.points[0][1]])
        self.init_map_x = norm(temp1 - temp2)

        temp1 = np.array([self.boundary_points.points[0][0], self.boundary_points.points[0][1]])
        temp2 = np.array([self.boundary_points.points[0][0], self.boundary_points.points[2][1]])
        self.init_map_y = norm(temp1 - temp2)

        self.start_x = (self.boundary_points.points[0][0] + self.boundary_points.points[2][0]) * 0.5
        self.start_y = (self.boundary_points.points[0][1] + self.boundary_points.points[2][1]) * 0.5

        self.V.append(np.array([self.odom_points[0], self.odom_points[1]]))

    def reset(self):
        self.boundary_points.points = []
        self.points.points = []
        self.line.points = []
        self.V = []
        self.first_run = True
        self.rate.sleep()

    def detect(self):
        while self.start_signal:
            xr = np.random.uniform(-0.5, 0.5, 1) * self.init_map_x + self.start_x
            yr = np.random.uniform(-0.5, 0.5, 1) * self.init_map_y + self.start_y
            x_rand = np.array([xr[0], yr[0]])
            x_nearest = utils.nearest(self.V, x_rand)
            x_new = utils.steer(x_nearest, x_rand, self.eta)
            obstacle_free, stopped_point = utils.obstacle_free(x_nearest, x_new, self.occupy_map_data)
            if obstacle_free == -1:  # unknown
                # p = Point()
                # p.x = stopped_point[0]
                # p.y = stopped_point[1]
                # p.z = 0.0
                # self.points.points.append(p)
                # self.shapes_pub.publish(self.points)
                # self.points.points = []

                goal = PointStamped()
                goal.header.stamp = rospy.Time.now()
                goal.header.frame_id = self.occupy_map_data.header.frame_id
                goal.point.x = stopped_point[0]
                goal.point.y = stopped_point[1]
                goal.point.z = 0.0

                self.detected_points_pub.publish(goal)

                if self.mode == 'local':
                    self.V = []
                    query_ok = 0
                    position = None
                    while query_ok == 0:
                        try:
                            position, _ = self.tf_listener.lookupTransform(self.map_frame, self.robot_frame, rospy.Time(0))
                            query_ok = 1
                        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                            query_ok = 0

                    self.V.append(np.array([position[0], position[1]]))
                    self.line.points = []
                    self.shapes_pub.publish(self.line)
                    # rospy.sleep(0.5)

            elif obstacle_free == 1:  # free
                self.V.append(stopped_point)
                position = Point()
                position.x = stopped_point[0]
                position.y = stopped_point[1]
                position.z = 0.0
                self.line.points.append(position)

                position = Point()
                position.x = x_nearest[0]
                position.y = x_nearest[1]
                position.z = 0.0
                self.line.points.append(position)
                self.shapes_pub.publish(self.line)
            # rospy.sleep(2)
            self.rate.sleep()

    def map_callback(self, data):
        self.occupy_map_data = data

    def boundary_points_callback(self, data):
        temp_point = Point()
        temp_point.x = data.point.x
        temp_point.y = data.point.y
        temp_point.z = data.point.z
        self.boundary_points.points.append(temp_point)

    def start_signal_callback(self, data):
        self.start_signal = data.data

    def reset_signal_callBack(self, data):
        self.reset_signal = data.data

    def setup_boundary_points(self):
        map_resolution = self.occupy_map_data.info.resolution
        map_width = self.occupy_map_data.info.height
        map_height = self.occupy_map_data.info.width
        pos_origin_x = self.occupy_map_data.info.origin.position.x
        pos_origin_y = self.occupy_map_data.info.origin.position.y

        self.boundary_points.points.append(np.array([((map_width * map_resolution) * -1.0) - pos_origin_x + 0.01,
                                                     ((map_height * map_resolution) * 1.0) + pos_origin_y - 0.01]))
        self.boundary_points.points.append(np.array([((map_width * map_resolution) * -1.0) - pos_origin_x + 0.01,
                                                     ((map_height * map_resolution) * -1.0) - pos_origin_y + 0.01]))
        self.boundary_points.points.append(np.array([((map_width * map_resolution) * 1.0) + pos_origin_x - 0.01,
                                                     ((map_height * map_resolution) * -1.0) - pos_origin_y + 0.01]))
        self.boundary_points.points.append(np.array([((map_width * map_resolution) * 1.0) + pos_origin_x - 0.01,
                                                     ((map_height * map_resolution) * 1.0) + pos_origin_y - 0.01]))

    def odom_callback(self, data):
        self.odom_points = [data.pose.pose.position.x, data.pose.pose.position.y]

    def init_visual_info(self):
        point_rgb = None
        line_rgb = None
        if self.mode == 'global':
            point_rgb = [0, 0, 1]  # blue point
            line_rgb = [9.0 / 255.0, 91.0 / 255.0, 236.0 / 255.0]  # shallow blue
        elif self.mode == 'local':
            point_rgb = [1, 0, 0]  # blue point
            line_rgb = [1, 0, 0]  # red line

        self.points.header.frame_id = self.occupy_map_data.header.frame_id
        self.points.header.stamp = rospy.Time.now()
        self.points.ns = self.mode + "_RRT"
        self.points.id = 0
        self.points.type = self.points.POINTS
        self.points.action = self.points.ADD
        self.points.pose.orientation.w = 1.0
        self.points.scale.x = 0.3
        self.points.scale.y = 0.3
        self.points.color.r = point_rgb[0]
        self.points.color.g = point_rgb[1]
        self.points.color.b = point_rgb[2]
        self.points.color.a = 0.5
        self.points.lifetime = rospy.Duration()

        self.line.header.frame_id = self.occupy_map_data.header.frame_id
        self.line.header.stamp = rospy.Time.now()
        self.line.ns = self.mode + "_RRT"
        self.line.id = 1
        self.line.type = self.line.LINE_LIST
        self.line.action = self.line.ADD
        self.line.pose.orientation.w = 1.0
        self.line.scale.x = 0.07
        self.line.scale.y = 0.07
        self.line.color.r = line_rgb[0]
        self.line.color.g = line_rgb[1]
        self.line.color.b = line_rgb[2]
        self.line.color.a = 0.2
        self.line.lifetime = rospy.Duration()
