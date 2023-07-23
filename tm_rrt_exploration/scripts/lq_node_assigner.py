import rospy

from lq_assigner import Assigner

if __name__ == '__main__':
    try:
        rn = 'tb3_0'
        rospy.init_node('lq_assigner', anonymous=False)
        assigner = Assigner(rn)
        assigner.init()
        assigner.loop()
    except rospy.ROSInterruptException:
        pass
