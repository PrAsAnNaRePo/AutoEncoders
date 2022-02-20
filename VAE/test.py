
import torch
from model import AutoEncoder, optim
from data import Data
import config
from torch.utils.data import DataLoader
from utils import train, save_checkpoint, testin_iter

data = Data(IMG_DIR='D:/Spider/image-segmentation/jpeg_images/IMAGES')

Autoencoder = AutoEncoder().to(config.DEVICE)
Autoencoder.load_state_dict(torch.load('30-4437.848-AutoEncoder.pth.tar')['model_state_dict'])
optim.load_state_dict(torch.load('30-4437.848-AutoEncoder.pth.tar')['optimizer'])

testin_iter(data, Autoencoder)
