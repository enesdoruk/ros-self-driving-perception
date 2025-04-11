#!/usr/bin/python
# -*- coding: UTF-8 -*-

import rospy    
import cv2
from sensor_msgs.msg import Image
import sys
import numpy as np
from rospy.numpy_msg import numpy_msg

import argparse
import logging
import os

import torch
import torch.nn.functional as F
from PIL import Image as pilImage
from torchvision import transforms

from Segmentation.unet import UNet
from Segmentation.data_vis import plot_img_and_mask
from Segmentation.dataset import BasicDataset

import time

prev_frame_time = 0
new_frame_time = 0

SegmentTalker = rospy.Publisher('Segmentation', Image, queue_size=10)


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold

def get_args():
  
    parser.add_argument('--model', '-m', default='/home/carf/carf_ROS/src/compVision/scripts/Segmentation/checkpoints/CP_epoch21.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=True)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return pilImage.fromarray((mask * 255).astype(np.uint8))

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


def runSegmentation(ros_goruntu):
    global prev_frame_time 
    global new_frame_time

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model ")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)

    net.load_state_dict(torch.load('/home/carf/carf_ROS/src/compVision/scripts/Segmentation/checkpoints/CP_epoch21.pth', map_location=device))

    new_frame_time = time.time()

    im = np.frombuffer(ros_goruntu.data, dtype=np.uint8).reshape(ros_goruntu.height, ros_goruntu.width, -1)
    RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = pilImage.fromarray(RGB)

    mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=0.5,
                           out_threshold=0.5,
                           device=device)

    result = mask_to_image(mask)
    resultcv = np.array(result)

    RGB = cv2.resize(RGB, (640, 480))
    resultcv = cv2.resize(resultcv, (640, 480))

    y1 = 0
    y2 = 0
    y = 0

    for i in range(resultcv.shape[0]):
        for j in range(resultcv.shape[1]):
            if (resultcv[i][j] > 10):
                y1 = j
                break
        for k in range(resultcv.shape[1]):
            if (resultcv[i][(resultcv.shape[1] -1) - k] > 10):
                y2 = (resultcv.shape[1] -1) - k
                break
        if (abs(y1 - y2) > 10):
            y = (y1 + y2) // 2
            resultcv[i][y] = 255
            resultcv[i-1][y] = 255
            resultcv[i+1][y] = 255
            resultcv[i-2][y] = 255
            resultcv[i+2][y] = 255

    #labeled_coins, _ = ndi.label(resultcv)
    #image_label_overlay = label2rgb(labeled_coins, image=resultcv, bg_label=-1)
    #image_label_overlay = image_label_overlay.astype('uint8') * 255
    #image_label_overlay_gray = cv2.cvtColor(image_label_overlay, cv2.COLOR_BGR2GRAY)
       

    resultcvRGB = cv2.cvtColor(resultcv, cv2.COLOR_GRAY2RGB)

    WCol1 = (255,255,255)
    WCol2= (50,50,50)
    mask = cv2.inRange(resultcvRGB, WCol2, WCol1)
    resultcvRGB[mask>0] = (255,0,0)

    combine = cv2.add(resultcvRGB, RGB)

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    cv2.putText(RGB, 'FPS: {}'.format(fps), (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    imgx = cv2_to_imgmsg(combine)
    SegmentTalker.publish(imgx)

    cv2.imshow("ObjSegment", combine)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('kapatiliyor...')


def main(args):
    rospy.init_node('talkerSeg', anonymous=True)
    rate = rospy.Rate(10)
    rospy.Subscriber("/camera/color/image_raw", Image, runSegmentation)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("kapatiliyor")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
