# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 18:19:52 2022

@author: prasa
"""
import os
import config
import PIL
import PIL.Image as Image
import albumentations
import numpy as np
import tensorflow as tf
import torch
import torchvision.models.resnet as resnet
from torch import nn, sigmoid, tensor
from torch.backends import cudnn
from torch.nn.functional import relu, leaky_relu
from torch.utils.data import DataLoader, Dataset
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models.efficientnet import efficientnet_b7

encoder = resnet.resnet152(pretrained=True)
for parameters in encoder.parameters():
    parameters.requires_grad = False
encoder.fc = nn.Linear(in_features=encoder.fc.in_features, out_features=12 * 12 * 10)


class Upsample(nn.Module):
    def __init__(self, size, in_f, out_f, **kwargs):
        super(Upsample, self).__init__()
        self.ups = nn.Sequential(
            nn.Upsample(size=size),
            nn.Conv2d(in_f, out_f, kernel_size=(3, 3), **kwargs)
        )

    def forward(self, x):
        return relu(self.ups(x))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.feat_inp = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=(3, 3)),
        )

        # self.img_inp = alexnet(pretrained=True)
        self.img_inp = efficientnet_b7(pretrained=True)

        for i in self.img_inp.parameters():
            i.requires_grad = False
        self.img_inp.classifier[1] = nn.Linear(2560, 12 * 12 * 10)

        self.Upscaling = nn.Sequential(
            Upsample((20, 20), 32, 64, padding=1),
            Upsample((60, 60), 64, 128, padding=1),
            Upsample((80, 80), 128, 32, padding=1),
            Upsample((130, 130), 32, 64, padding=1),
            Upsample((150, 150), 64, 128, padding=1),
            Upsample((180, 180), 128, 32, padding=1),
            Upsample((200, 200), 32, 64, padding=1),
            Upsample((210, 210), 64, 128, padding=1),
            Upsample((220, 220), 128, 3, padding=1),

        )

    def forward(self, features, img):
        f, im = self.feat_inp(features.reshape(-1, 10, 12, 12)), self.feat_inp(
            self.img_inp(img).reshape(-1, 10, 12, 12))
        combo_ = f + im
        x = self.Upscaling(combo_)
        return relu(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder()

    def forward(self, img):
        f = self.encoder(img)
        return relu(self.decoder(f, img))


# reconstruction_loss = tf.reduce_mean(
#                 keras.losses.binary_crossentropy(data, reconstruction)
#             )

model = AutoEncoder().to(config.DEVICE)
loss = nn.MSELoss()
optim = torch.optim.Adamax(model.parameters(), lr=config.LEARNING_RATE)
lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, verbose=True)

