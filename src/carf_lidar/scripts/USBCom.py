#!/usr/bin/python
# -*- coding: UTF-8 -*-

import rospy
from std_msgs.msg import Float32MultiArray
import serial
import time

def sendData(data):
    angle = int(data.data[0])
    speed = int(data.data[1])

    if angle > 100:
        print(" sola ", (200 - angle), " derece ")
        print(" Hız:  ", speed, " m/sn")
    elif angle < 100:
        print(" saga ", angle, " derece ")
        print(" Hız:  ", speed, " m/sn")

    ser = serial.Serial('/dev/ttyS0', 115200)
    
    arrayAngle = bytearray([angle, speed])
   
    ser.write(arrayAngle)
    ser.close()

def usb_data():
    rospy.init_node('usb_node', anonymous=True)
    rospy.Subscriber("/usb_com", Float32MultiArray, sendData)
    rospy.spin()

if __name__ == '__main__':
    usb_data()
