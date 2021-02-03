from network.net_build import *

import logging
import os
import sys
from glob import glob

import torch.nn as nn
from torch import optim
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

dir_img = 'data/imgs/'
dir_real = 'data/real/'
dir_checkpoint = 'checkpoints/'
data_num = 0

def train(
          G,
          D,
          device,
          epochs=1,
          batch_size=1,
          lr=0.001,
          val_percent=0.1,
          save_cp=True,
          img_scale=0.5):
    
    train_data=Data_set(dir_img, 0.5)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(epochs):
        G.train()
        D_blur.train()
        D_gray.train()

        for batch in train_loader:
            fake=batch['fake']
            real=batch['real']
            
            num+=1
            size=fake.size(0)

            real_label=torch.ones(size)
            fake_lable=torch.zeros(size)

            
            fake=fake.to(device=device, dtype=torch.float32)
            real=real.to(device=device, dtype=torch.float32)
            real_label=real_label.to(device=device, dtype=torch.float32)
            fake_label=fake_label.to(device=device, dtype=torch.float32)

            fake_out=G(fake).detach()
            fake_output=guide_filter()

            fake_blur=guild_filter(fake_output, fake_output)#Part 1. Blur GAN
            real_blur=guild_filter(real, real)

            fake_gray=color_shift(fake_output)#Part 2.Gray GAN
            real_gray=color_shift(real)

            fake_disc_blur=D_blur(fake_blur)
            real_disc_blur=D_blur(real_blur)
            fake_disc_gray=D_gray(fake_gray)
            real_disc_gray=D_gray(real_gray)

            loss_real_blur=criterion(real_disc_blur, real_label)
            loss_fake_blur=criterion(fake_disc_blur, fake_label)
            loss_real_gray=criterion(real_disc_gray, real_label)
            loss_fake_gray=criterion(fake_disc_gray, fake_label)

            loss_blur=loss_real_blur+loss_fake_blur
            loss_gray=loss_real_gray+loss_fake_gray

            D_blur_optimizer.zero_grad()
            D_gray_optimizer.zero_grad()
            loss_blur.backward()
            loss_gray.backward()
            D_blur_optimizer.step()
            D_gray_optimizer.step()

            fake_out=G(fake)
            fake_output=guide_filter()
            fake_blur=guild_filter(fake_output, fake_output)
            fake_disc_blur=D_blur(fake_blur)
            loss_fake_blur=criterion(fake_disc_blur, fake_label)

            G_optimizer.zero_grad()
            loss_fake_blur.backward()
            G_optimizer.step()

            fake_out=G(fake)
            fake_output=guide_filter()
            fake_gray=color_shift(fake_output)
            fake_disc_gray=D_gray(fake_gray)
            loss_fake_gray=criterion(fake_disc_gray, fake_label)
            G_optimizer.zero_grad()
            loss_fake_gray.backward()
            G_optimizer.step()


            VGG1=VGG(fake)
            VGG2=VGG(fake_output)

            superpixel_img=SuperPixel(fake)

            VGG3=VGG(superpixel_img)

            loss1=Get_loss(VGG1, VGG3)
            loss2=Get_loss(VGG2, VGG3)

            loss_sum=loss1+loss2

            G_optimizer.zero_grad()
            loss_sum.backward()
            G_optimizer.step()




