# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 18:21:59 2022

@author: prasa
"""

import random
from matplotlib import pyplot as plt
import torch
import config
class CustomToTensor:
    def __call__(self, data, shape):
        return torch.from_numpy(data).reshape(shape=shape)
        

def train(epochs, train_loader, model, loss, optimizer, lr_sch=None, checkpoint=None, **kwargs):
    print('started training...')
    model.train()
    for e in range(1, epochs+1):
        for batch, (x, y) in enumerate(train_loader):
            pred = model(x.to(config.DEVICE))
            l = loss(pred, y.to(config.DEVICE))

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        lr_sch.step(l, e) if lr_sch else ()
        print(f'epoch {e}/{epochs} with the loss {l.item():.3f}')
        if checkpoint:
            checkpoint(e, epochs,l, model, optimizer, **kwargs)


def save_checkpoint(e, epoch, loss, model, optimizer, Bloss = 0.0299, show_img=False, directry='./', name='None', sample=None):
    if show_img and e%5==0:
        model.eval()
        pred = model(sample.to(config.DEVICE).reshape(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[0]))
        plt.imshow(pred.cpu().detach().numpy().reshape(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3))
        plt.show()
    if loss < Bloss:
        print('saving.....')
        ckpnt = {'model_state_dict': model.state_dict(),
                'architecture': model,
                'optimizer': optimizer.state_dict()}
        torch.save(ckpnt, f'{directry}{e}-{loss:.3f}-{name}.pth.tar')
    
    if epoch==e:
        print('saving.....')
        ckpnt = {'model_state_dict': model.state_dict(),
                'architecture': model,
                'optimizer': optimizer.state_dict()}
        torch.save(ckpnt, f'{directry}{e}-{loss:.3f}-{name}.pth.tar')



def testin_iter(label, model):
    sample = label[random.randint(1, 100)][0]
    pred = model(sample.to(config.DEVICE).reshape(1, 3, 220, 220))
    plt.imshow(pred.cpu().detach().numpy().reshape(220, 220, 3))
    plt.show()