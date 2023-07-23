import rospy

from lq_filter import Filter

if __name__ == '__main__':
    try:
        rn = 'tb3_0'
        nn = 'lq_filter'
        rospy.init_node(nn, anonymous=False)

        frontier_filter = Filter(rn)
        frontier_filter.init()
        frontier_filter.loop()
    except rospy.ROSInterruptException:
        pass
