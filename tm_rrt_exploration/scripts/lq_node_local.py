import numpy as np
import rospy

from lq_detector import Detector

if __name__ == '__main__':
    try:
        rn = 'tb3_0'
        np.random.seed(42)
        nn = 'lq_local_rrt_detector'
        rospy.init_node(nn, anonymous=False)
        ld = Detector('local', rn)
        ld.init()
        ld.loop()
    except rospy.ROSInterruptException:
        pass
