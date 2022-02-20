from operator import mod
from model import model, loss, optim, lr_sch
from data import Data
import config
from torch.utils.data import DataLoader
from utils import train, save_checkpoint
import GPUtil
GPUtil.showUtilization()
data = Data(IMG_DIR='D:/Spider/image-segmentation/jpeg_images/IMAGES')
train_data = DataLoader(data, batch_size=config.BATCH_SIZE)
train(config.EPOCHS,
        train_loader=train_data,
        model=model,
        loss=loss,
        optimizer=optim,
        lr_sch=lr_sch,
        checkpoint=save_checkpoint,
        name='AutoEncoder',
        )
