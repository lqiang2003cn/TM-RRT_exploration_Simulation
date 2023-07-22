#!/usr/bin/env python

import random
import time
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
        self.robot_goal_cancel = None
        self.time_interval_due = None
        self.next_time_interval = None
        self.next_assign_time = None
        self.temp_inv_array = None
        self.map_data = OccupancyGrid()
        self.frontiers = []
        self.invalid_frontier = []
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
        self.inv_frontier_topic = None
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

    def init(self):
        self.map_topic = rospy.get_param('~map_topic', '/map')
        self.info_radius = rospy.get_param('~info_radius', 1.0)
        self.info_multiplier = rospy.get_param('~info_multiplier', 3.0)
        self.hysteresis_radius = rospy.get_param('~hysteresis_radius', 3.0)
        self.hysteresis_gain = rospy.get_param('~hysteresis_gain', 2.0)
        self.frontiers_topic = rospy.get_param('~frontiers_topic', '/filtered_points')
        self.inv_points_topic = rospy.get_param('~invalid_frontier', '/invalid_points')
        self.inv_frontier_topic = rospy.get_param('~invalid_centroids', '/invalid_centroids')
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

        points = Marker()
        points.header.frame_id = self.map_data.header.frame_id
        points.header.stamp = rospy.Time.now()
        points.ns = "invalidCentroid"
        points.id = 0
        points.type = Marker.POINTS
        points.action = Marker.ADD
        points.pose.orientation.w = 1.0
        points.scale.x = 0.4
        points.scale.y = 0.4
        points.color.r = 255.0 / 255.0
        points.color.g = 0.0 / 255.0
        points.color.b = 255.0 / 255.0
        points.color.a = 0.5
        points.lifetime = rospy.Duration()

        rospy.sleep(self.start_delay)

    def reset(self):
        # reinitialize everything.
        self.robot_assigned_goal = []
        self.temp_inv_array = []
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
        centroids = copy(self.frontiers)

        if self.next_time_interval <= rospy.get_rostime().secs:
            self.time_interval_due = True
            self.next_time_interval = rospy.get_rostime().secs + self.message_time_interval

        invalid_goal_array = invalidArray()
        invalid_goal_array.points = []

        for inv_frt in range(0, len(self.invalid_frontier)):
            tempPoint = Point()
            tempPoint.z = 0.0
            tempPoint.x = self.invalid_frontier[inv_frt][0]
            tempPoint.y = self.invalid_frontier[inv_frt][1]
            invalid_goal_array.points.append(copy(tempPoint))

        movebase_status = self.robot.get_movebase_status()
        # wrong status, cancel goal
        if movebase_status >= 4 and self.robot_assigned_goal['time_thres'] != -1:
            distance = norm(self.robot.get_position(), self.robot_assigned_goal['goal'])
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

        centroid_record = []
        robot_position = self.robot.get_position()
        robot_goal = self.robot_assigned_goal['goal']

        if self.robot_goal_cancel:
            is_available = True
        else:
            if self.robot.get_state() == 1:
                is_available = False
            else:
                is_available = True

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

            revenue = info_gain_ip - cost
            centroid_record.append(centroids[ip])

        if len(centroids) > 0:
            if rospy.get_rostime().secs >= self.next_assign_time:
                not_assign = True
                attempt = 0
                start = time.time()
                skip_assign = False
                if not is_available:
                    goal_distance = norm(self.robot.get_position(), self.robot_assigned_goal['goal'])
                    time_elapsed = rospy.get_rostime().secs - self.robot_assigned_goal['time_start']
                    if self.hysteresis_radius <= goal_distance <= (self.rp_metric_distance * 1.5):
                        skip_assign = True
                    if time_elapsed > self.non_interrupt_time and goal_distance <= self.hysteresis_radius:
                        skip_assign = True

                if not skip_assign:
                    while not_assign and attempt < len(centroids):
                        winner_id = getHigestIndex(revenue_record[xxx], attempt)
                        if winner_id != -1:
                            # condition for the general robot
                            cond_goal = True
                            cond_history = True
                            cond_goalTaken = True
                            cond_busyNear = True
                            for xi in range(0, len(invalidFrontier)):
                                if calculateLocationDistance(centroid_record[xxx][winner_id],
                                                             invalidFrontier[xi]) < 0.1:
                                    cond_goal = False
                                    if debugFlag1:
                                        print(">> invalid goal detected. distance: %.3f" % (
                                            calculateLocationDistance(centroid_record[xxx][winner_id],
                                                                      invalidFrontier[xi])))
                                    break
                            # check if history appear in other robot history
                            history = []
                            for xj in range(0, len(robot_namelist)):
                                history.append(robots[xj].getGoalHistory())
                            for xj in range(0, len(robot_namelist)):
                                if xxx != xj:
                                    for xh in range(0, len(history[xj])):
                                        distance_apart = calculateLocationDistance(
                                            centroid_record[xxx][winner_id], history[xj][xh])
                                        if distance_apart <= 0.3:
                                            cond_goalTaken = False
                                            if debugFlag1:
                                                print(
                                                        'Goal [%f,%f] for robot %s repeated at other robot %s history' % (
                                                    centroid_record[xxx][winner_id][0],
                                                    centroid_record[xxx][winner_id][1], robot_namelist[xxx],
                                                    robot_namelist[xj]))
                                            break
                                # if the goal is repeated at own history
                                else:
                                    for xh in range(0, len(history[xj])):
                                        distance_apart = calculateLocationDistance(
                                            centroid_record[xxx][winner_id], history[xj][xh])
                                        if distance_apart <= 0.1:
                                            remainingTime = ((robot_assigned_goal[xj]['time_start'] +
                                                              robot_assigned_goal[xj][
                                                                  'time_thres']) - rospy.get_rostime().secs)
                                            if remainingTime < (0.0 - delay_after_assignment):
                                                cond_history = False
                                                if debugFlag1:
                                                    print(
                                                            'history repeat the same goal assignment in robot %s history' % (
                                                        robot_namelist[xxx]))
                                                break
                            if xxx in nb:
                                distance_apart = calculateLocationDistance(centroid_record[xxx][winner_id],
                                                                           robot_assigned_goal[xxx]['goal'])
                                if distance_apart >= (hysteresis_radius * 2.0):
                                    cond_busyNear = False

                            # checking the condition for the goal assignment
                            if cond_history and cond_goal and cond_goalTaken and cond_busyNear:
                                position3, rot3 = robots[xxx].getPosition(quad=True)
                                robots[xxx].send_goal(point=centroid_record[xxx][winner_id], quadData=rot3)
                                robots[xxx].setGoalHistory(centroid_record[xxx][winner_id])
                                robot_assigned_goal[xxx]['lastgoal'] = robot_assigned_goal[xxx]['goal'].copy()
                                robot_assigned_goal[xxx]['goal'] = centroid_record[xxx][winner_id]
                                robot_assigned_goal[xxx]['startLoc'] = position3
                                robot_assigned_goal[xxx]['time_start'] = rospy.get_rostime().secs
                                robot_assigned_goal[xxx]['time_thres'] = calcDynamicTimeThreshold(position3,
                                                                                                  centroid_record[
                                                                                                      xxx][
                                                                                                      winner_id],
                                                                                                  time_per_meter)
                                rospy.loginfo("\n" + robot_namelist[xxx] + " has been assigned to  " + str(
                                    centroid_record[xxx][winner_id]) +
                                              " mission start at " + str(
                                    robot_assigned_goal[xxx]['time_start']) + " sec  - Limit: " + str(
                                    int(robot_assigned_goal[xxx]['time_thres'])) + "s")
                                not_assign = False
                            else:
                                # if not assigned
                                attempt = attempt + 1
                                if debugFlag2:
                                    print("======== Robot: %s attempt no %d to assign goal ==========" % (
                                        robot_namelist[xxx], attempt))
                        else:
                            if debugFlag1:
                                print("++++++== Robot: %s give up assigning goal after %d attempt ==++++++" % (
                                    robot_namelist[xxx], attempt))

                    if debugFlag1:
                        print('    Assignment of goal for robot %s used %.6f seconds \n' % (
                            robot_namelist[xxx], (time.time() - start)))

                    # check whether there is invalid goal or history after numerous attempt
                    if (cond_history or cond_goalTaken) and not cond_goal:
                        goal_repeated = True
                        faultyGoal = centroid_record[xxx][winner_id]
                        if goal_repeated:
                            repeat_invalid = False
                            for ie in range(0, len(invalidFrontier)):
                                if calculateLocationDistance(invalidFrontier[ie],
                                                             robot_assigned_goal[xxx]['goal']) < 0.01:
                                    repeat_invalid = True
                            if repeat_invalid == False:
                                tempInvGoal = Point()
                                tempInvGoal.x = robot_assigned_goal[xxx]['goal'][0]
                                tempInvGoal.y = robot_assigned_goal[xxx]['goal'][1]
                                tempInvGoal.z = 0.0
                                # if the distance is very far then do not publish as invalid point, mostly cant reach goal within the given time limit
                                distance = calculateLocationDistance(robots[xxx].getPosition(),
                                                                     robot_assigned_goal[xxx]['goal'])
                                invalid_goal_array.points.append(copy(tempInvGoal))
                                invalidFrontier.append(robot_assigned_goal[xxx]['goal'])
                                temp_inv_array.append(robot_assigned_goal[xxx]['goal'])
                            if debugFlag1:
                                print('Robot: %d assigned goal is repeated in the history' % (winner_id))
                            robot_assigned_goal[xxx]['lastgoal'] = faultyGoal
                            robot_assigned_goal[xxx]['goal'] = robots[xxx].getPosition()
                            robot_assigned_goal[xxx]['time_start'] = rospy.get_rostime().secs
                            robot_assigned_goal[xxx]['startLoc'] = robots[xxx].getPosition()
                            robot_assigned_goal[xxx]['time_thres'] = -1
                            # cancel goal
                            rospy.loginfo("\n !!!> Robot " + robot_namelist[xxx] + " give up goal: " + str(
                                robot_assigned_goal[xxx]['goal'])
                                          + " at time: " + str(int(curr_time)) + " sec  -- distance: " + str(
                                distance) + " -- goal repeated \n")
                            robots[xxx].cancelGoal()
                            robot_goal_cancel.append(xxx)
                rospy.sleep(0.1)

                next_assign_time = rospy.get_rostime().secs + delay_after_assignment

        # -------------------------------------------------------------------------
        # check if the robot assignment time out.
        for ix in range(0, len(robot_namelist)):
            curr_time = rospy.get_rostime().secs
            if robot_assigned_goal[ix]['time_thres'] != -1:
                if (((robot_assigned_goal[ix]['time_start'] + robot_assigned_goal[ix][
                    'time_thres']) - curr_time) <= 0):
                    # for for repeated invalid frontier
                    repeat_invalid = False
                    distance = calculateLocationDistance(robots[ix].getPosition(), robot_assigned_goal[ix]['goal'])
                    for ie in range(0, len(invalidFrontier)):
                        if calculateLocationDistance(invalidFrontier[ie], robot_assigned_goal[ix]['goal']) < 0.1:
                            repeat_invalid = True
                    if repeat_invalid == False:
                        tempInvGoal = Point()
                        tempInvGoal.x = robot_assigned_goal[ix]['goal'][0]
                        tempInvGoal.y = robot_assigned_goal[ix]['goal'][1]
                        tempInvGoal.z = 0.0

                        # if the distance is very far then do not publish as invalid point, mostly cant reach goal within the given time limit
                        if distance <= invalid_distance:
                            invalid_goal_array.points.append(copy(tempInvGoal))
                            invalidFrontier.append(robot_assigned_goal[ix]['goal'])
                            temp_inv_array.append(robot_assigned_goal[ix]['goal'])

                    # need to reset the assigned tasks to the robot
                    robot_assigned_goal[ix]['lastgoal'] = robot_assigned_goal[ix]['goal'].copy()
                    robot_assigned_goal[ix]['goal'] = robots[ix].getPosition()
                    robot_assigned_goal[ix]['time_start'] = rospy.get_rostime().secs
                    robot_assigned_goal[ix]['startLoc'] = robots[ix].getPosition()
                    robot_assigned_goal[ix]['time_thres'] = -1
                    # cancel goal
                    rospy.loginfo("!!!> Robot " + robot_namelist[ix] + " give up goal: " + str(
                        robot_assigned_goal[ix]['goal'])
                                  + " at time: " + str(curr_time) + " sec  -- distance: " + str(
                        distance) + " -- goal repeated")
                    robots[ix].cancelGoal()
                    robot_goal_cancel.append(ix)

        # -------------------------------------------------------------------------
        # wont print so frequent
        if time_interval_due:
            rospy.loginfo('publishing invalid goals :' + str(temp_inv_array))
            # print out the distance traveled by each of the robot
            for ii in range(0, len(robot_namelist)):
                print("Robot - %d   total distance traveled: %.2fm" % (ii, robots[ii].getDistanceTraveled()))

        # publish invalid location for the points
        invPub.publish(invalid_goal_array)
        points.points = copy(invalid_goal_array.points)
        invCenPub.publish(points)
        time_interval_due = False

    def cancel(self):
        # cancel the robot goal if the signal is terminated
        for ji in range(0, len(robot_namelist)):
            robots[ji].cancelGoal()
            robot_assigned_goal[ji]['lastgoal'] = robot_assigned_goal[ji]['goal'].copy()
            robot_assigned_goal[ji]['goal'] = robots[ji].getPosition()
            robot_assigned_goal[ji]['time_start'] = rospy.get_rostime().secs
            robot_assigned_goal[ji]['startLoc'] = robots[ji].getPosition()
            robot_assigned_goal[ji]['time_thres'] = -1
        rospy.loginfo('-----------------+++ Assigner Module: Terminated Signal Received +++----------------- ')
        rospy.sleep(2.0)

    def loop(self):
        temp_inv_array = []
        next_assign_time = rospy.get_rostime().secs
        next_time_interval = rospy.get_rostime().secs
        time_interval_due = False
        robot_goal_cancel = []
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


if __name__ == '__main__':
    try:
        rn = 'tb3_0'
        rospy.init_node('assigner', anonymous=False)
        assigner = Assigner(rn)
        assigner.init()
        assigner.loop()
    except rospy.ROSInterruptException:
        pass
