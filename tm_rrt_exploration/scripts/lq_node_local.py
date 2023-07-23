import numpy as np
import rospy

from lq_detector import Detector

if __name__ == '__main__':
    try:
        np.random.seed(42)
        nn = 'lq_local_rrt_detector'
        rospy.init_node(nn, anonymous=False)
        ld = Detector(nn, 'local', 'tb3_0')
        ld.init()
        ld.loop()
    except rospy.ROSInterruptException:
        pass
