#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy    
import cv2
from sensor_msgs.msg import Image
import sys
import numpy as np
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage


def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding")
    dtype = np.dtype("uint8") 
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), dtype=dtype, buffer=img_msg.data) 
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height 
    return img_msg

def SegmentListener(ros_goruntu):
    print("Segmentation Alindi")
    imgseg = imgmsg_to_cv2(ros_goruntu)

    cv2.imshow("Listener_Segment", imgseg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('kapatiliyor...')

def DetectListener(ros_goruntu):
    print("Detection Alindi")
    imgdtct = imgmsg_to_cv2(ros_goruntu)

    cv2.imshow("Listener_Detect", imgdtct)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('kapatiliyor...')
    
def main(args):
    rospy.init_node('CARF_CAM', anonymous=True)
    rospy.Subscriber("/Segmentation", Image, SegmentListener)

    rospy.Subscriber("/Detection", Image, DetectListener)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("kapatiliyor")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)