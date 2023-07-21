import numpy as np
import rospy
from lq_detector import Detector

if __name__ == '__main__':
    try:
        np.random.seed(42)
        nn = 'global_rrt_detector'
        rospy.init_node(nn, anonymous=False)
        gd = Detector(nn, 'global')
        gd.init()
        gd.loop()
    except rospy.ROSInterruptException:
        pass
