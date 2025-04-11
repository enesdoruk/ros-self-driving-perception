import random
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
import cv2
 
class Augmentation:
    def __init__(self):
        pass
    def rotate(self,image,mask,angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-180, 180]) # -180~180 randomly choose an angle to rotate
        if isinstance(angle,list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
    def flip(self,image,mask): #Horizontal flip and vertical flip
        if random.random()>0.5:
            image = tf.hflip(image)
            mask = tf.hflip(mask)
        if random.random()<0.5:
            image = tf.vflip(image)
            mask = tf.vflip(mask)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
    def randomResizeCrop(self,image,mask,scale=(0.3,1.0),ratio=(1,1)):#scale indicates that the randomly cropped image will be between 0.3 and 1 times, and ratio indicates the aspect ratio
        img = np.array(image)
        h_image, w_image = img.shape[0], img.shape[1]
        resize_size = h_image
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
    def adjustContrast(self,image,mask):
        factor = transforms.RandomRotation.get_params([0,10]) #here the contrast of the expanded data
        image = tf.adjust_contrast(image,factor)
        mask = tf.adjust_contrast(mask,factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image,mask
    def adjustBrightness(self,image,mask):
        factor = transforms.RandomRotation.get_params([1, 2]) #Here adjust the brightness of the data after expansion
        image = tf.adjust_brightness(image, factor)
        mask = tf.adjust_contrast(mask, factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
    def centerCrop(self,image,mask): #Center crop
        image = tf.center_crop(image,(640, 360))
        mask = tf.center_crop(mask,(640, 360))
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image,mask
    def adjustSaturation(self,image,mask): #Adjust saturation
        factor = transforms.RandomRotation.get_params([1, 2]) # Adjust the brightness of the data after expansion
        image = tf.adjust_saturation(image, factor)
        mask = tf.adjust_saturation(mask, factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
 
 
def augmentationData(image_path,mask_path,option=[1,2,3,4,5,6,7],save_dir=None):
    '''
         :param image_path: the path of the image
         :param mask_path: path of the mask
         :param option: Which augmentation method is needed: 1 is rotation, 2 is flipping, 3 is random cropping and restoring the original size, 4 is adjusting contrast, 5 is center cropping (not restoring the original size), 6 is adjusting brightness, 7 Is saturation
         :param save_dir: The path where the augmented data is stored
    '''
    aug_image_savedDir = os.path.join(save_dir,'img')
    aug_mask_savedDir = os.path.join(save_dir, 'mask')
    if not os.path.exists(aug_image_savedDir):
        os.makedirs(aug_image_savedDir)
        print('create aug image dir.....')
    if not os.path.exists(aug_mask_savedDir):
        os.makedirs(aug_mask_savedDir)
        print('create aug mask dir.....')
    aug = Augmentation()
    res= os.walk(image_path)
    images = []
    masks = []
    for root,dirs,files in res:
        for f in files:
            images.append(os.path.join(root,f))
    res = os.walk(mask_path)
    for root,dirs,files in res:
        for f in files:
            masks.append(os.path.join(root,f))
    datas = list(zip(images,masks))
    num = len(datas)
 
    for (image_path,mask_path) in datas:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        if 1 in option:
            num+=1
            image_tensor, mask_tensor = aug.rotate(image, mask)
            image_rotate = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_rotate.jpg'))
            mask_rotate = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_rotate.jpg'))
        if 2 in option:
            num+=1
            image_tensor, mask_tensor = aug.flip(image, mask)
            image_filp = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir,'img',str(num)+'_flip.jpg'))
            mask_filp = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir,'mask',str(num)+'_flip.jpg'))
        if 3 in option:
            num+=1
            image_tensor, mask_tensor = aug.randomResizeCrop(image, mask)
            image_ResizeCrop = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_ResizeCrop.jpg'))
            mask_ResizeCrop = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_ResizeCrop.jpg'))
        if 4 in option:
            num+=1
            image_tensor, mask_tensor = aug.adjustContrast(image, mask)
            image_Contrast = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_Contrast.jpg'))
            mask_Contrast = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_Contrast.jpg'))
        if 5 in option:
            num+=1
            image_tensor, mask_tensor = aug.centerCrop(image, mask)
            image_centerCrop = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_centerCrop.jpg'))
            mask_centerCrop = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_centerCrop.jpg'))
        if 6 in option:
            num+=1
            image_tensor, mask_tensor = aug.adjustBrightness(image, mask)
            image_Brightness = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_Brightness.jpg'))
            mask_Brightness = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_Brightness.jpg'))
        if 7 in option:
            num+=1
            image_tensor, mask_tensor = aug.adjustSaturation(image, mask)
            image_Saturation = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_Saturation.jpg'))
            mask_Saturation = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_Saturation.jpg'))
 
 
augmentationData('/home/enes/Desktop/BTU_IEEE_AI/LaneSegment/data/imgs','/home/enes/Desktop/BTU_IEEE_AI/LaneSegment/data/masks',save_dir='/home/enes/Desktop/BTU_IEEE_AI/LaneSegment/data/augs')