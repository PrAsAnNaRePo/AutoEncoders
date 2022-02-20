# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 18:21:27 2022

@author: prasa
"""
import os
import matplotlib.pyplot as plt
import PIL
import PIL.Image as Image
import config
import cv2
import albumentations
import torch
from utils import CustomToTensor
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


albumentation = albumentations.Compose([
    albumentations.RandomBrightnessContrast(p=0.7),
    albumentations.Blur(p=0.8),
    albumentations.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
    # albumentations.ColorJitter(saturation=0.4, hue=0.3, p=0.6),
    albumentations.RandomFog(p=0.7),
    albumentations.RandomRain(p=0.6),
    albumentations.RandomShadow(p=0.6),
])

class Data(Dataset):
    def __init__(self, IMG_DIR=None):
        self.img_dir = IMG_DIR
        self.IMAGES_PATH = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.IMAGES_PATH)

    def __getitem__(self, item):
        img = np.array(PIL.Image.open(f'{self.img_dir}/{self.IMAGES_PATH[item]}'))
        img = cv2.resize(img, config.IMAGE_SIZE)
        x_img = albumentation(image=img)
        x_img = x_img['image']
        x_img = torch.from_numpy(x_img).reshape(3, 220, 220)
        x_img = torch.tensor(x_img, dtype=torch.float32)
        y_img = torch.from_numpy(img).reshape(3, 220, 220)
        y_img = torch.tensor(y_img, dtype=torch.float32)
        return x_img, y_img

print('data loaded')
# data = Data(IMG_DIR='D:/Spider/image-segmentation/jpeg_images/IMAGES')
# print(data[0][0].shape)
# plt.imshow(data[0][0].reshape(220, 220, 3))
# plt.show()


    
    
    
    
    
    
    
    
    
    