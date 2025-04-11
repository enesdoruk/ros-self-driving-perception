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
from std_msgs.msg import Int32MultiArray
import message_filters

bridge = CvBridge()


def deptCam(image, coord):
    print('took depth image')
    global bridge

    x0 = coord.data[0]
    y0 = coord.data[1]
    x1 = coord.data[2]
    y1 = coord.data[3]

    print('x0 = ', x0, 'y0 = ', y0,'x1 = ', x1, 'y1 = ', y1)

    img = bridge.imgmsg_to_cv2(image)

    cv_image_array = np.array(img, dtype = np.dtype('f8'))
    cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX) 

    cv2.imshow("DepthKamera", cv_image_norm)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('kapatiliyor...')

    
def main(args):
    global bridge

    rospy.init_node('DepthKamera', anonymous=True)

    imagedepth = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
    coord = message_filters.Subscriber("/coordPub", Int32MultiArray)

    ts = message_filters.ApproximateTimeSynchronizer([imagedepth,coord],10,0.1,allow_headerless=True)
    ts.registerCallback(deptCam)


    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("kapatiliyor")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)