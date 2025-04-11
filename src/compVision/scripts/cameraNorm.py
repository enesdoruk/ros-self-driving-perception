#!/usr/bin/python3
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
VideoRaw = rospy.Publisher('VideoRaw', Image, queue_size=10)
VideoGray = rospy.Publisher('VideoGray', Image, queue_size=10)

def func(ros_goruntu):
    print('goruntu alindi')
    global bridge

    img = bridge.imgmsg_to_cv2(ros_goruntu, "bgr8")
    imgx = bridge.cv2_to_imgmsg(img)
    VideoRaw.publish(imgx)


    cv2.imshow("kamera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('kapatiliyor...')

def funcGray(ros_goruntu):
    print('Gray goruntu alindi')
    global bridge

    img = bridge.imgmsg_to_cv2(ros_goruntu, "bgr8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgx = bridge.cv2_to_imgmsg(img)
    VideoGray.publish(imgx)


    cv2.imshow("kameragray", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('kapatiliyor...')
    
def main(args):
    global bridge

    rospy.init_node('kamera', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, func)
    rospy.Subscriber("/camera/color/image_raw", Image, funcGray)
    
    print("python version: ", sys.version)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("kapatiliyor")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)