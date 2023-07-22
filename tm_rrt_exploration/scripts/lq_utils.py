import actionlib
import numpy as np
import rospy
import tf
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan
from numpy import array
from numpy.linalg import norm


def discount2(mapData, assigned_pt, centroids, infoGain, r):
    for j in range(0, len(infoGain)):
        temp_infoGain = 0.0
        index = index_of_point(mapData, centroids[j])
        r_region = int(r / mapData.info.resolution)
        init_index = index - r_region * (mapData.info.width + 1)
        for n in range(0, 2 * r_region + 1):
            start = n * mapData.info.width + init_index
            end = start + 2 * r_region
            limit = ((start / mapData.info.width) + 2) * mapData.info.width
            for i in range(start, end + 1):
                if 0 <= i < limit and i < len(mapData.data):
                    if (mapData.data[i] == -1 and norm(array(centroids[j]) - point_of_index(mapData, i)) <= r and norm(
                            point_of_index(mapData, i) - assigned_pt) <= r):
                        temp_infoGain += 1.0
        infoGain[j] -= (temp_infoGain * (mapData.info.resolution ** 2))
    return infoGain


def steer(x_nearest, x_rand, eta):
    x_new = np.zeros(2)
    if norm(x_nearest - x_rand) <= eta:
        x_new = x_rand
    else:
        m = (x_rand[1] - x_nearest[1]) / (x_rand[0] - x_nearest[0])
        x_new[0] = np.sign(x_rand[0] - x_nearest[0]) * np.sqrt(pow(eta, 2) / (pow(m, 2) + 1)) + x_nearest[0]
        x_new[1] = m * (x_new[0] - x_nearest[0]) + x_nearest[1]
        if x_rand[0] == x_nearest[0]:
            x_new[0] = x_nearest[0]
            x_new[1] = x_nearest[1] + eta
    return x_new


def relative_position_metric(inputLoc, robotIndex, robots_position, distance_threshold=3.0):
    if len(robots_position) > 1:
        inputList = []
        for i in range(0, len(robots_position)):
            if i != robotIndex:
                inputList.append(robots_position[i])
        # now calculate relative position for each of the robot in the list
        distanceList = []
        for j in inputList:
            distanceList.append(norm(j, inputLoc))
        # now get the largest distance and perform calculation on the data
        minDistance = min(distanceList)
        if minDistance > distance_threshold:
            return 1.0
        else:
            temp = minDistance / distance_threshold
            if temp < 0.01:
                return 0.01
            else:
                return temp
    else:
        return 1.0


def nearest(V, x):
    min_dist = np.inf
    min_v = None  # assuming V is not empty
    for v in V:
        dist = norm(v - x)
        if dist < min_dist:
            min_dist = dist
            min_v = v
    return min_v


def grid_value(map_data, point):
    resolution = map_data.info.resolution
    m_startx = map_data.info.origin.position.x
    m_starty = map_data.info.origin.position.y
    width = map_data.info.width
    idx = (np.floor((point[1] - m_starty) / resolution) * width) + np.floor((point[0] - m_startx) / resolution)
    return map_data.data[int(idx)]


def obstacle_free(x_nearest, x_new, map_data):
    rez = float(map_data.info.resolution) * 0.2
    steps = int(np.ceil(norm(x_new - x_nearest)) / rez)
    xi = x_nearest
    obs = 0
    unknown = 0

    for i in range(steps):
        xi = steer(xi, x_new, rez)
        grid_v = grid_value(map_data, xi)  # grid_v: 0:free, -1:unknown, 100:obstacle
        if grid_v > 80:
            obs = 1
        elif grid_v == -1:
            unknown = 1
            break

    # out: -1: unknown; 0: obstacle; 1: free
    if unknown == 1:
        out = -1
    elif obs == 1:
        out = 0
    else:
        out = 1
    return out, xi


def cal_dist(input_loc, dest_loc):
    return np.sqrt(np.power(input_loc[0] - dest_loc[0], 2) + np.power(input_loc[1] - dest_loc[1], 2))


def grid_value_merged_map(mapData, Xp, distance=2):
    resolution = mapData.info.resolution
    xstartx = mapData.info.origin.position.x
    xstarty = mapData.info.origin.position.y

    width = mapData.info.width
    Data = mapData.data
    # returns grid value at "Xp" location
    # map data:  100 occupied      -1 unknown       0 free
    index = np.floor((Xp[1] - xstarty) / resolution) * width + np.floor((Xp[0] - xstartx) / resolution)
    outData = square_area_check(Data, index, width, distance)
    if len(outData) > 1:
        if 100 not in outData:
            if max(outData, key=outData.count) == -1 and max(outData) == 0:
                return -1
            elif max(outData, key=outData.count) == -1 and max(outData) == -1 and 0 not in outData:
                return 100
            elif max(outData, key=outData.count) == -1 and max(outData) == -1 and 0 in outData:
                return -1
            elif max(outData, key=outData.count) == 0 and -1 in outData:
                return -1
            else:
                return -1
        else:
            return 100
    else:
        return 100


# ________________________________________________________________________________
def square_area_check(data, index, width, distance=2):
    # now using the data to perform a square area check on the data for removing the invalid point
    dataOutList = []
    for j in range(-1 * distance, distance + 1):
        for i in range(int(index) + ((width * j) - distance), int(index) + ((width * j) + distance)):
            if i < len(data):
                dataOutList.append(data[int(i)])
    return dataOutList


def index_of_point(mapData, Xp):
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y
    width = mapData.info.width
    index = int((np.floor((Xp[1] - Xstarty) / resolution) * width) + (np.floor((Xp[0] - Xstartx) / resolution)))
    return index


def point_of_index(mapData, i):
    y = mapData.info.origin.position.y + (i / mapData.info.width) * mapData.info.resolution
    x = mapData.info.origin.position.x + float(
        i - (int(i / mapData.info.width) * mapData.info.width)) * mapData.info.resolution
    # modified for certain python version might mismatch the int and float conversion
    return np.array([x, y])


def information_gain(mapData, point, r):
    infoGain = 0.0
    index = index_of_point(mapData, point)
    r_region = int(r / mapData.info.resolution)
    init_index = index - r_region * (mapData.info.width + 1)
    for n in range(0, 2 * r_region + 1):
        start = n * mapData.info.width + init_index
        end = start + 2 * r_region
        limit = ((start / mapData.info.width) + 2) * mapData.info.width
        for i in range(start, end + 1):
            if 0 <= i < limit and i < len(mapData.data):
                if mapData.data[i] == -1 and norm(np.array(point) - point_of_index(mapData, i)) <= r:
                    infoGain = infoGain + 1.0
    return infoGain * (mapData.info.resolution ** 2)


class Robot:
    def __init__(self, name, move_base_service, in_service, global_frame, base_link):
        rospy.loginfo('setting up robot init for robot - ' + name)
        self.assigned_point = []
        self.goal_history = []
        self.name = name
        self.goal = MoveBaseGoal()
        self.start = PoseStamped()
        self.end = PoseStamped()
        self.global_frame = rospy.get_param('~global_frame', global_frame)
        self.robot_frame = rospy.get_param('~robot_frame', base_link)
        self.plan_service = rospy.get_param('~plan_service', in_service)
        self.listener = tf.TransformListener()
        self.listener.waitForTransform(self.global_frame, self.name + '/' + self.robot_frame, rospy.Time(0), rospy.Duration(10))
        self.total_distance = 0.0
        self.first_run = True
        self.movebase_status = 0
        self.sub = rospy.Subscriber(name + "/odom", Odometry, self.odom_callback)
        self.status_sub = rospy.Subscriber(name + "/move_base/status", GoalStatusArray, self.movebase_status_callback)
        cond = 0
        pos = None
        while cond == 0:
            try:
                rospy.loginfo('Waiting for the robot transform')
                (pos, rot) = self.listener.lookupTransform(self.global_frame, self.name + '/' + self.robot_frame, rospy.Time(0))
                cond = 1
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.sleep(0.1)
                cond = 0
        self.position = array([pos[0], pos[1]])
        self.previous_x = 0
        self.previous_y = 0
        self.assigned_point = self.position
        self.client = actionlib.SimpleActionClient('/' + self.name + move_base_service, MoveBaseAction)
        self.client.wait_for_server()
        self.goal.target_pose.header.frame_id = self.global_frame
        self.goal.target_pose.header.stamp = rospy.Time.now()
        rospy.wait_for_service(self.name + self.plan_service)
        self.make_plan = rospy.ServiceProxy(self.name + self.plan_service, GetPlan)
        self.start.header.frame_id = self.global_frame
        self.end.header.frame_id = self.global_frame

    def odom_callback(self, data):
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y

        if self.first_run:
            self.previous_x = x
            self.previous_y = y
        d_increment = np.sqrt((x - self.previous_x) * (x - self.previous_x) + (y - self.previous_y) * (y - self.previous_y))
        self.total_distance = self.total_distance + d_increment
        self.first_run = False
        self.previous_x = x
        self.previous_y = y

    def movebase_status_callback(self, data):
        if len(data.status_list) > 0:
            self.movebase_status = max([status.status for status in data.status_list])

    def get_movebase_status(self):
        return self.movebase_status

    def get_distance_traveled(self):
        return self.total_distance

    def get_position(self, quad=False):
        cond = 0
        pos, rot = None, None
        while cond == 0:
            try:
                (pos, rot) = self.listener.lookupTransform(self.global_frame, self.name + '/' + self.robot_frame, rospy.Time(0))
                cond = 1
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                cond = 0
        self.position = array([pos[0], pos[1]])

        if not quad:
            return self.position
        else:
            return self.position, rot

    def transform_point_to_robot_frame(self, trans):
        while not rospy.is_shutdown():
            try:
                point = PoseStamped()
                point.header.frame_id = self.global_frame
                point.pose.position.x = trans[0]
                point.pose.position.y = trans[1]
                point.pose.orientation.w = 1.0
                transformed = self.listener.transformPose(self.name + '/' + self.robot_frame, point)
                return transformed.pose.position.x, transformed.pose.position.y
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

    def send_goal_transformed(self, point):
        transform_point = self.transform_point_to_robot_frame(point)
        self.goal.target_pose.pose.position.x = transform_point[0]
        self.goal.target_pose.pose.position.y = transform_point[1]
        self.goal.target_pose.pose.orientation.w = 1.0
        self.client.send_goal(self.goal)
        self.goal_history.append(np.array(point))
        self.assigned_point = np.array(point)

    def send_goal(self, point, quadData=None):
        if quadData is None:
            self.goal.target_pose.pose.position.x = point[0]
            self.goal.target_pose.pose.position.y = point[1]
            self.goal.target_pose.pose.orientation.w = 1.0
        else:
            self.goal.target_pose.pose.position.x = point[0]
            self.goal.target_pose.pose.position.y = point[1]
            self.goal.target_pose.pose.orientation.x = quadData[0]
            self.goal.target_pose.pose.orientation.y = quadData[1]
            self.goal.target_pose.pose.orientation.z = quadData[2]
            self.goal.target_pose.pose.orientation.w = quadData[3]
        self.client.send_goal(self.goal)
        self.assigned_point = array(point)

    def cancel_goal(self, noQuadFlag=False):
        if noQuadFlag:
            point = self.get_position()
            self.goal.target_pose.pose.position.x = point[0]
            self.goal.target_pose.pose.position.y = point[1]
            self.goal.target_pose.pose.orientation.w = 1.0
        else:
            point, quadData = self.get_position(quad=True)
            self.goal.target_pose.pose.position.x = point[0]
            self.goal.target_pose.pose.position.y = point[1]
            self.goal.target_pose.pose.orientation.x = quadData[0]
            self.goal.target_pose.pose.orientation.y = quadData[1]
            self.goal.target_pose.pose.orientation.z = quadData[2]
            self.goal.target_pose.pose.orientation.w = quadData[3]
        # send the goal
        self.client.send_goal(self.goal)

    def get_state(self):
        return self.client.get_state()

    def make_plan(self, start, end):
        self.start.pose.position.x = start[0]
        self.start.pose.position.y = start[1]
        self.end.pose.position.x = end[0]
        self.end.pose.position.y = end[1]
        start = self.listener.transformPose(self.name + '/map', self.start)
        end = self.listener.transformPose(self.name + '/map', self.end)
        plan = self.make_plan(start=start, goal=end, tolerance=0.2)
        return plan.plan.poses

    def set_goal_history(self, point):
        self.goal_history.append(point)

    def get_goal_history(self):
        return self.goal_history
