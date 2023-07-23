#!/usr/bin/env python

from copy import copy

# --------Include modules---------------
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from numpy import array
from numpy.linalg import norm
from std_msgs.msg import Bool
# import math
from tmrrt_exploration.msg import PointArray, invalidArray
from visualization_msgs.msg import Marker

import lq_utils as utils


class Assigner:

    def __init__(self, robot_name):
        self.invalid_frontiers_shape = None
        self.invalid_frontiers_shape_pub = None
        self.invalid_frontiers_pub = None
        self.invalid_frontiers_shape_topic = None
        self.invalid_frontiers_topic = None
        self.robot_goal_cancel = None
        self.time_interval_due = None
        self.next_time_interval = None
        self.next_assign_time = None
        self.map_data = OccupancyGrid()
        self.frontiers = []
        self.invalid_frontiers = []
        self.global_maps = []
        self.start_signal = True
        self.reset_signal = False
        self.map_topic = None
        self.info_radius = None
        self.info_multiplier = None
        self.hysteresis_radius = None
        self.hysteresis_gain = None
        self.frontiers_topic = None
        self.inv_points_topic = None
        self.inv_centroids_topic = None
        self.time_per_meter = None
        self.message_time_interval = None
        self.delay_after_assignment = None
        self.invalid_distance = None
        self.rp_metric_distance = None
        self.non_interrupt_time = None
        self.start_delay = None
        self.rate_hz = None
        self.start_signal_topic = None
        self.reset_signal_topic = None
        self.global_frame = None
        self.plan_service = None
        self.base_link = None
        self.move_base_service = None
        self.rate = None
        self.robot_assigned_goal = {}
        self.robot = None
        self.robot_name = robot_name
        self.start_time = None
        self.rp_metric = 1.0
        self.robot_goal_cancel = None

    def init(self):
        self.map_topic = rospy.get_param('~map_topic', self.robot_name + '/map')
        self.info_radius = rospy.get_param('~info_radius', 1.0)
        self.info_multiplier = rospy.get_param('~info_multiplier', 3.0)
        self.hysteresis_radius = rospy.get_param('~hysteresis_radius', 3.0)
        self.hysteresis_gain = rospy.get_param('~hysteresis_gain', 2.0)
        self.frontiers_topic = rospy.get_param('~frontiers_topic', '/filtered_points')
        self.invalid_frontiers_topic = rospy.get_param('~invalid_frontiers', '/invalid_frontiers')
        self.invalid_frontiers_shape_topic = rospy.get_param('~invalid_frontiers_shape', '/invalid_frontiers_shape')
        self.time_per_meter = rospy.get_param('~time_per_meter', 12)
        self.message_time_interval = rospy.get_param('~message_time_interval', 2.0)
        self.delay_after_assignment = rospy.get_param('~delay_after_assignment', 1.0)
        self.invalid_distance = rospy.get_param('~invalid_distance', 1.0)
        self.rp_metric_distance = rospy.get_param('~rp_metric_distance', 10.0)
        self.non_interrupt_time = rospy.get_param('~non_interrupt_time', 1.2)
        self.start_delay = rospy.get_param('~start_delay', 1.0)
        self.rate_hz = rospy.get_param('~rate', 100)
        self.start_signal_topic = rospy.get_param('~startSignalTopic', '/explore_start')
        self.reset_signal_topic = rospy.get_param('~resetSignalTopic', '/explore_reset')
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.plan_service = rospy.get_param('~plan_service', '/move_base_node/NavfnROS/make_plan')
        self.base_link = rospy.get_param('~base_link', 'base_footprint')
        self.move_base_service = rospy.get_param('~move_base_service', '/move_base')
        self.rate = rospy.Rate(self.rate_hz)

        # -------------------------------------------
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_call_back)
        rospy.Subscriber(self.frontiers_topic, PointArray, self.frontiers_call_back)
        rospy.Subscriber(self.start_signal_topic, Bool, self.start_signal_call_back)
        rospy.Subscriber(self.reset_signal_topic, Bool, self.reset_signal_call_back)

        while len(self.frontiers) < 1:
            pass

        while len(self.map_data.data) < 1:
            pass

        self.robot = self.create_robot()

        pos, rot = self.robot.get_position(quad=True)
        self.robot.send_goal(point=pos, quadData=rot)
        self.start_time = rospy.get_rostime().secs
        self.robot_assigned_goal = {
            'goal': pos,
            'startLoc': pos,
            'time_start': self.start_time,
            'valid': True,
            'time_thres': -1,
            'lastgoal': pos
        }

        self.invalid_frontiers_pub = rospy.Publisher(self.invalid_frontiers_topic, invalidArray, queue_size=10)
        self.invalid_frontiers_shape_pub = rospy.Publisher(self.invalid_frontiers_shape_topic, Marker, queue_size=10)

        self.invalid_frontiers_shape = Marker()
        self.invalid_frontiers_shape.header.frame_id = self.map_data.header.frame_id
        self.invalid_frontiers_shape.header.stamp = rospy.Time.now()
        self.invalid_frontiers_shape.ns = "invalidCentroid"
        self.invalid_frontiers_shape.id = 0
        self.invalid_frontiers_shape.type = Marker.POINTS
        self.invalid_frontiers_shape.action = Marker.ADD
        self.invalid_frontiers_shape.pose.orientation.w = 1.0
        self.invalid_frontiers_shape.scale.x = 0.4
        self.invalid_frontiers_shape.scale.y = 0.4
        self.invalid_frontiers_shape.color.r = 255.0 / 255.0
        self.invalid_frontiers_shape.color.g = 0.0 / 255.0
        self.invalid_frontiers_shape.color.b = 255.0 / 255.0
        self.invalid_frontiers_shape.color.a = 0.5
        self.invalid_frontiers_shape.lifetime = rospy.Duration()

        rospy.sleep(self.start_delay)

    def reset(self):
        # reinitialize everything.
        self.robot_assigned_goal = []
        self.next_assign_time = rospy.get_rostime().secs
        self.next_time_interval = rospy.get_rostime().secs
        self.time_interval_due = False
        self.robot_goal_cancel = False
        self.robot = self.create_robot()
        pos, rot = self.robot.get_position(quad=True)
        self.robot.send_goal(point=pos, quadData=rot)
        self.robot_assigned_goal = {
            'goal': pos,
            'startLoc': pos,
            'time_start': self.start_time,
            'valid': True,
            'time_thres': -1,
            'lastgoal': pos
        }

    def do_assign(self):
        self.next_time_interval = rospy.get_rostime().secs
        self.next_assign_time = rospy.get_rostime().secs
        centroids = copy(self.frontiers)
        if self.next_time_interval <= rospy.get_rostime().secs:
            self.time_interval_due = True
            self.next_time_interval = rospy.get_rostime().secs + self.message_time_interval

        invalid_goal_array = invalidArray()
        invalid_goal_array.points = []

        for inv_frt in range(0, len(self.invalid_frontiers)):
            tempPoint = Point()
            tempPoint.z = 0.0
            tempPoint.x = self.invalid_frontiers[inv_frt][0]
            tempPoint.y = self.invalid_frontiers[inv_frt][1]
            invalid_goal_array.points.append(copy(tempPoint))

        movebase_status = self.robot.get_movebase_status()
        # wrong status, cancel goal
        if movebase_status >= 4 and self.robot_assigned_goal['time_thres'] != -1:
            self.robot_assigned_goal['lastgoal'] = self.robot_assigned_goal['goal']
            self.robot_assigned_goal['goal'] = self.robot.get_position()
            self.robot_assigned_goal['time_start'] = rospy.get_rostime().secs
            self.robot_assigned_goal['startLoc'] = self.robot.get_position()
            self.robot_assigned_goal['time_thres'] = -1
            self.robot.cancel_goal()

        # start
        self.robot_goal_cancel = True
        info_gain = []
        for ip in range(0, len(centroids)):
            ig = utils.information_gain(self.map_data, [centroids[ip][0], centroids[ip][1]], self.info_radius)
            info_gain.append(ig * self.info_multiplier)
        info_gain = utils.discount2(self.map_data, self.robot_assigned_goal['lastgoal'], centroids, info_gain, self.info_radius)

        robot_position = self.robot.get_position()
        robot_goal = self.robot_assigned_goal['goal']

        if self.robot_goal_cancel:
            is_available = True
        else:
            if self.robot.get_state() == 1:
                is_available = False
            else:
                is_available = True

        revenue_record = []
        centroid_record = []
        for ip in range(0, len(centroids)):
            cost = norm(robot_position - centroids[ip])
            info_gain_ip = info_gain[ip]
            if not is_available:
                if norm(centroids[ip] - robot_position) <= self.hysteresis_radius:
                    info_gain_ip *= self.hysteresis_gain
                if norm(robot_goal - centroids[ip]) <= self.hysteresis_radius:
                    ig = utils.information_gain(self.map_data, [centroids[ip][0], centroids[ip][1]], self.info_radius)
                    info_gain_ip = ig * self.hysteresis_gain
            else:
                if norm(centroids[ip] - robot_position) <= self.hysteresis_radius:
                    info_gain_ip *= self.hysteresis_gain

            if info_gain_ip >= 0:
                info_gain_ip = info_gain_ip * self.rp_metric
            else:
                info_gain_ip = info_gain_ip / self.rp_metric

            revenue_record.append(info_gain_ip - cost)
            centroid_record.append(centroids[ip])

        if len(centroids) > 0 and rospy.get_rostime().secs >= self.next_assign_time:
            not_assign = True
            attempt = 0
            skip_assign = False
            if not is_available:
                goal_distance = norm(self.robot.get_position() - self.robot_assigned_goal['goal'])
                time_elapsed = rospy.get_rostime().secs - self.robot_assigned_goal['time_start']
                if self.hysteresis_radius <= goal_distance <= (self.rp_metric_distance * 1.5):
                    skip_assign = True
                if time_elapsed > self.non_interrupt_time and goal_distance <= self.hysteresis_radius:
                    skip_assign = True

            if not skip_assign:
                valid = True
                not_repeated_goal = True
                winner_id = 0

                while not_assign and attempt < len(centroids):
                    winner_id = utils.get_highest_index(revenue_record, attempt)
                    # condition for the general robot
                    valid = True
                    not_repeated_goal = True
                    near_curr_goal = True

                    for inv_frt in range(0, len(self.invalid_frontiers)):
                        if norm(centroid_record[winner_id] - self.invalid_frontiers[inv_frt]) < 0.1:
                            valid = False
                            break

                    history = self.robot.get_goal_history()
                    for hist in range(0, len(history)):
                        distance_apart = norm(centroid_record[winner_id] - history[hist])
                        if distance_apart <= 0.1:
                            finish_time = self.robot_assigned_goal['time_start'] + self.robot_assigned_goal['time_thres']
                            # if current goal is out of time, then remaining_time < 0
                            # if current goal is in time, then remaining_time >= 0
                            remaining_time = finish_time - rospy.get_rostime().secs
                            # if current goal is out of time, then count it as a repeated goal
                            if remaining_time < (0.0 - self.delay_after_assignment):
                                not_repeated_goal = False
                                break

                    if not is_available:
                        distance_apart = norm(centroid_record[winner_id] - self.robot_assigned_goal['goal'])
                        if distance_apart >= self.hysteresis_radius * 2.0:  # too far away from current goal
                            near_curr_goal = False

                    # checking the condition for the goal assignment
                    if valid and not_repeated_goal and near_curr_goal:
                        pos, rot = self.robot.get_position(quad=True)
                        self.robot.send_goal(point=centroid_record[winner_id], quadData=rot)
                        self.robot.set_goal_history(centroid_record[winner_id])
                        self.robot_assigned_goal['lastgoal'] = self.robot_assigned_goal['goal'].copy()
                        self.robot_assigned_goal['goal'] = centroid_record[winner_id]
                        self.robot_assigned_goal['startLoc'] = pos
                        self.robot_assigned_goal['time_start'] = rospy.get_rostime().secs
                        dyn_time_thre = utils.calc_dyn_time_thre(pos, centroid_record[winner_id], self.time_per_meter)
                        self.robot_assigned_goal['time_thres'] = dyn_time_thre
                        not_assign = False
                    else:
                        attempt = attempt + 1

                # not repeated goal but invalid
                if not_repeated_goal and not valid:
                    faulty_goal = centroid_record[winner_id]
                    repeat_invalid = False

                    # make the invalid standard lower
                    for ie in range(0, len(self.invalid_frontiers)):
                        if norm(self.invalid_frontiers[ie] - self.robot_assigned_goal['goal']) < 0.01:
                            repeat_invalid = True

                    if not repeat_invalid:
                        temp_inv_goal = Point()
                        temp_inv_goal.x = self.robot_assigned_goal['goal'][0]
                        temp_inv_goal.y = self.robot_assigned_goal['goal'][1]
                        temp_inv_goal.z = 0.0
                        invalid_goal_array.points.append(copy(temp_inv_goal))
                        self.invalid_frontiers.append(self.robot_assigned_goal['goal'])

                    self.robot_assigned_goal['lastgoal'] = faulty_goal
                    self.robot_assigned_goal['goal'] = self.robot.get_position()
                    self.robot_assigned_goal['time_start'] = rospy.get_rostime().secs
                    self.robot_assigned_goal['startLoc'] = self.robot.get_position()
                    self.robot_assigned_goal['time_thres'] = -1
                    self.robot.cancel_goal()
                    self.robot_goal_cancel = True

            self.rate.sleep()
            self.next_assign_time = rospy.get_rostime().secs + self.delay_after_assignment

        # -------------------------------------------------------------------------
        # check if the robot assignment time out.
        curr_time = rospy.get_rostime().secs
        if self.robot_assigned_goal['time_thres'] != -1:
            # if goal is not finished in time:
            if self.robot_assigned_goal['time_start'] + self.robot_assigned_goal['time_thres'] - curr_time <= 0:
                repeat_invalid = False
                distance = norm(self.robot.get_position() - self.robot_assigned_goal['goal'])
                for ie in range(0, len(self.invalid_frontiers)):
                    if norm(self.invalid_frontiers[ie] - self.robot_assigned_goal['goal']) < 0.1:
                        repeat_invalid = True

                if not repeat_invalid:
                    temp_inv_goal = Point()
                    temp_inv_goal.x = self.robot_assigned_goal['goal'][0]
                    temp_inv_goal.y = self.robot_assigned_goal['goal'][1]
                    temp_inv_goal.z = 0.0

                    if distance <= self.invalid_distance:
                        invalid_goal_array.points.append(copy(temp_inv_goal))
                        self.invalid_frontiers.append(self.robot_assigned_goal['goal'])

                # need to reset the assigned tasks to the robot
                self.robot_assigned_goal['lastgoal'] = self.robot_assigned_goal['goal'].copy()
                self.robot_assigned_goal['goal'] = self.robot.get_position()
                self.robot_assigned_goal['time_start'] = rospy.get_rostime().secs
                self.robot_assigned_goal['startLoc'] = self.robot.get_position()
                self.robot_assigned_goal['time_thres'] = -1
                # cancel goal
                self.robot.cancel_goal()
                self.robot_goal_cancel = True

        # publish invalid location for the points
        self.invalid_frontiers_pub.publish(invalid_goal_array)
        self.invalid_frontiers_shape.points = copy(invalid_goal_array.points)
        self.invalid_frontiers_shape_pub.publish(self.invalid_frontiers_shape)
        self.rate.sleep()

    def cancel(self):
        self.robot.cancel_goal()
        self.robot_assigned_goal['lastgoal'] = self.robot_assigned_goal['goal'].copy()
        self.robot_assigned_goal['goal'] = self.robot.get_position()
        self.robot_assigned_goal['time_start'] = rospy.get_rostime().secs
        self.robot_assigned_goal['startLoc'] = self.robot.get_position()
        self.robot_assigned_goal['time_thres'] = -1
        rospy.sleep(2.0)

    def loop(self):
        while not rospy.is_shutdown():
            if self.reset_signal:
                self.reset()
            else:
                if self.start_signal:
                    self.do_assign()
                else:
                    self.cancel()
            self.rate.sleep()

    def frontiers_call_back(self, data):
        self.frontiers = []
        for point in data.points:
            self.frontiers.append(array([point.x, point.y]))

    def map_call_back(self, data):
        self.map_data = data

    def start_signal_call_back(self, data):
        self.start_signal = data.data

    def reset_signal_call_back(self, data):
        self.reset_signal = data.data

    def create_robot(self):
        return utils.Robot(
            name=self.robot_name,
            move_base_service=self.move_base_service,
            in_service=self.plan_service,
            global_frame=self.global_frame,
            base_link=self.base_link
        )
