import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, Point
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from numpy.linalg import norm
import lq_utils as utils


class Detector:

    def __init__(self, node_name, mode):
        self.start_x = None
        self.start_y = None
        self.init_map_x = None
        self.init_map_y = None
        self.start_signal = False
        self.reset_signal = False
        self.occupy_map_data = OccupancyGrid()
        self.boundary_points = Marker()
        self.line = Marker()
        self.points = Marker()
        self.first_run = True
        self.V = []
        self.eta = 1
        self.map_topic = '/map'
        self.rate_hz = 100
        self.detected_points_pub = None
        self.shapes_pub = None
        self.node_name = node_name
        self.rate = None
        self.mode = mode

    def init_visual_info(self):
        self.points.header.frame_id = self.occupy_map_data.header.frame_id
        self.points.header.stamp = rospy.Time.now()
        self.points.ns = "global_RRT"
        self.points.id = 0
        self.points.type = self.points.POINTS
        self.points.action = self.points.ADD
        self.points.pose.orientation.w = 1.0
        self.points.scale.x = 0.3
        self.points.scale.y = 0.3
        self.points.color.r = 255.0 / 255.0
        self.points.color.g = 0.0 / 255.0
        self.points.color.b = 0.0 / 255.0
        self.points.color.a = 0.5
        self.points.lifetime = rospy.Duration()

        self.line.header.frame_id = self.occupy_map_data.header.frame_id
        self.line.header.stamp = rospy.Time.now()
        self.line.ns = "global_RRT"
        self.line.id = 1
        self.line.type = self.line.LINE_LIST
        self.line.action = self.line.ADD
        self.line.pose.orientation.w = 1.0
        self.line.scale.x = 0.07
        self.line.scale.y = 0.07
        self.line.color.r = 9.0 / 255.0
        self.line.color.g = 91.0 / 255.0
        self.line.color.b = 236.0 / 255.0
        self.line.color.a = 0.2
        self.line.lifetime = rospy.Duration()

    def cal_init_info(self):
        temp1 = np.array([self.points.points[0].x, self.points.points[0].y])
        temp2 = np.array([self.points.points[2].x, self.points.points[0].y])
        self.init_map_x = norm(temp1 - temp2)

        temp1 = np.array([self.points.points[0].x, self.points.points[0].y])
        temp2 = np.array([self.points.points[0].x, self.points.points[2].y])
        self.init_map_y = norm(temp1 - temp2)

        self.start_x = (self.points.points[0].x + self.points.points[2].x) * 0.5
        self.start_y = (self.points.points[0].y + self.points.points[2].y) * 0.5

        trans = self.points.points[4]
        self.V.append(np.array([trans.x, trans.y]))

    def reset(self):
        self.boundary_points.points = []
        self.points.points = []
        self.line.points = []
        self.V = []
        self.first_run = True

    def explore(self):
        xr = np.random.uniform(-0.5, 0.5, 1) * self.init_map_x + self.start_x
        yr = np.random.uniform(-0.5, 0.5, 1) * self.init_map_y + self.start_y
        x_rand = np.array([xr, yr])
        x_nearest = utils.nearest(self.V, x_rand)
        x_new = utils.steer(x_nearest, x_rand, self.eta)
        obstacle_free, stopped_point = utils.obstacle_free(x_nearest, x_new, self.occupy_map_data)
        if obstacle_free == -1:
            p = Point()
            p.x = stopped_point[0]
            p.y = stopped_point[1]
            p.z = 0.0
            self.points.points.append(p)
            self.shapes_pub.publish(p)

            goal = PointStamped()
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = self.occupy_map_data.header.frame_id
            goal.point.x = stopped_point[0]
            goal.point.y = stopped_point[1]
            goal.point.z = 0.0
            self.detected_points_pub.publish(goal)
            self.points.points = []
        elif obstacle_free == 1:
            self.V.append(stopped_point)
            p = Point()
            p.x = stopped_point[0]
            p.y = stopped_point[1]
            p.z = 0.0
            self.line.points.append(p)

            p = Point()
            p.x = x_nearest[0]
            p.y = x_nearest[1]
            p.z = 0.0
            self.line.points.append(p)
        self.shapes_pub.publish(self.line)

    def map_callback(self, data):
        self.occupy_map_data = data

    def rviz_callback(self, data):
        temp_point = Point()
        temp_point.x = data.point.x
        temp_point.y = data.point.y
        temp_point.z = data.point.z
        self.boundary_points.points.append(temp_point)

    def start_signal_callback(self, data):
        self.start_signal = data.data

    def reset_signal_callBack(self, data):
        self.reset_signal = data.data

    def init(self):
        self.eta = rospy.get_param('~eta', 0.5)
        self.map_topic = rospy.get_param('~map_topic', "/map")
        self.rate_hz = rospy.get_param('~rate', 100)

        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback, queue_size=100)
        rospy.Subscriber("/clicked_point", PointStamped, self.rviz_callback, queue_size=100)
        rospy.Subscriber("/explore_start", Bool, self.start_signal_callback, queue_size=100)
        rospy.Subscriber("/explore_reset", Bool, self.reset_signal_callBack, queue_size=100)

        self.detected_points_pub = rospy.Publisher('/detected_points', PointStamped, queue_size=10)
        self.shapes_pub = rospy.Publisher(self.node_name + '/_shapes', Marker, queue_size=10)

        self.rate = rospy.Rate(self.rate_hz)

        while len(self.occupy_map_data.data) < 1:
            rospy.loginfo('Waiting for the map')
            rospy.sleep(0.1)
            pass

    def loop(self):
        while not rospy.is_shutdown():
            if self.reset_signal:
                self.reset()
                self.rate.sleep()
            else:
                if self.start_signal:
                    if self.first_run:
                        self.init_visual_info()
                        while self.points.points.size() < 5:
                            self.rate.sleep()
                        self.cal_init_info()
                        self.shapes_pub.publish(self.points)
                        self.first_run = False

                    if not self.first_run:
                        while self.start_signal:
                            self.explore()
                            self.rate.sleep()
                else:
                    self.rate.sleep()


if __name__ == '__main__':
    try:
        nn = 'global_rrt_detector'
        rospy.init_node(nn, anonymous=False)
        gd = Detector(nn, 'global')
        gd.init()
        gd.loop()

        # ld = Detector(nn, 'local')
        # ld.init()
        # ld.loop()
    except rospy.ROSInterruptException:
        pass
