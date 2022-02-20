# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 18:21:46 2022

@author: prasa
"""
from torch.backends import cudnn

DEVICE = 'cuda'
IMAGE_SIZE = (220, 220) 
LEARNING_RATE = 0.001
LATENT_SIZE = (10, 12, 12)
BATCH_SIZE = 12
EPOCHS = 30
cudnn.benchmark = True