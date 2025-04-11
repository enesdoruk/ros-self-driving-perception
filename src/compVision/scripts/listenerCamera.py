#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import rospy    
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
import numpy as np
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

bridge = CvBridge()


def func(ros_goruntu):
    print('goruntu alindi')
    global bridge

    img = bridge.imgmsg_to_cv2(ros_goruntu)

    cv2.imshow("Listener_kamera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('kapatiliyor...')

    
def main(args):
    global bridge

    rospy.init_node('ListenerKamera', anonymous=True)
    rospy.Subscriber("/VideoRaw", Image, func)
    

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("kapatiliyor")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)