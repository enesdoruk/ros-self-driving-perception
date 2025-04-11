#!/usr/bin/python
# -*- coding: UTF-8 -*-

import rospy    
import cv2
from sensor_msgs.msg import Image
import sys
import numpy as np
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Int32MultiArray

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np


from ObjDetect.models.experimental import attempt_load
from ObjDetect.utils.datasets import LoadStreams, LoadImages, letterbox
from ObjDetect.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from ObjDetect.utils.plots import plot_one_box
from ObjDetect.utils.torch_utils import select_device, load_classifier, time_synchronized

prev_frame_time = 0
new_frame_time = 0


DetectTalker = rospy.Publisher('Detection', Image, queue_size=10)
coordPub = rospy.Publisher('coordPub', Int32MultiArray, queue_size=10)



def ObjDetection(frame, save_img=False):
    weights, view_img,  imgsz = '/home/carf/carf_ROS/src/CompVision/scripts/ObjDetect/weights/best.pt', 1, 640
    
    set_logging()
    device = select_device('0')
    half = device.type != 'cpu'  

    model = attempt_load(weights, map_location=device)  
    imgsz = check_img_size(imgsz, s=model.stride.max()) 
    if half:
        model.half()  
    
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  
    _ = model(img.half() if half else img) if device.type != 'cpu' else None 

    img = letterbox(frame, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0 

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img, augment=True)[0]

    pred = non_max_suppression(pred, 0.7, 0.8, agnostic=True)
    t2 = time_synchronized()

    return img,pred,names,colors

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


def runDetection(ros_goruntu):
    global prev_frame_time 
    global new_frame_time

    new_frame_time = time.time()

    coord = []

    im = np.frombuffer(ros_goruntu.data, dtype=np.uint8).reshape(ros_goruntu.height, ros_goruntu.width, -1)
    RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    img,pred,names,colors = ObjDetection(RGB)
    for i, det in enumerate(pred):
        s, im0 =  '', RGB
        s += '%gx%g ' % img.shape[2:]  
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):  
                label = f'{names[int(cls)]}'
                plot_one_box(xyxy, im0, label=label, color=(255,0,0), line_thickness=1)

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    cv2.putText(im0, 'FPS: {}'.format(fps), (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    
    imgx = cv2_to_imgmsg(im0) 
    DetectTalker.publish(imgx)

    x0,y0,x1,y1 = 50,50,200,200
    coord.append(x0)
    coord.append(x1)
    coord.append(y0)
    coord.append(y1)

    coord = Int32MultiArray(data=coord)
    coordPub.publish(coord)

    cv2.imshow("ObjDetection", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('kapatiliyor...')


def main(args):
    
    rospy.init_node('talkerDetect', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, runDetection)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("kapatiliyor")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
