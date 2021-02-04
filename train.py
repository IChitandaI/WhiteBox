from network.net_build import *
from superpix import slic, adaptive_slic, sscolor

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
from torch.distributions import Distribution

dir_img = 'data/imgs/'
dir_real = 'data/real/'
dir_checkpoint = 'checkpoints/'
data_num = 0

def simple_superpixel(batch_image: np.ndarray, superpixel_fn: callable) -> np.ndarray:
  """ convert batch image to superpixel
  Args:
      batch_image (np.ndarray): np.ndarry, shape must be [b,h,w,c]
      seg_num (int, optional): . Defaults to 200.
  Returns:
      np.ndarray: superpixel array, shape = [b,h,w,c]
  """
  num_job = batch_image.shape[0]
  batch_out = Parallel(n_jobs=num_job)(delayed(superpixel_fn)
                                       (image) for image in batch_image)
  return np.array(batch_out)

class ColorShift(nn.Module):
  def __init__(self, mode='uniform'):
    super().__init__()
    self.dist: Distribution = None
    self.mode = mode

  def setup(self, device: torch.device):
    # NOTE 原论文输入的bgr图像，此处需要改为rgb
    if self.mode == 'normal':
      self.dist = torch.distributions.Normal(
          torch.tensor((0.299, 0.587, 0.114), device=device),
          torch.tensor((0.1, 0.1, 0.1), device=device))
    elif self.mode == 'uniform':
      self.dist = torch.distributions.Uniform(
          torch.tensor((0.199, 0.487, 0.014), device=device),
          torch.tensor((0.399, 0.687, 0.214), device=device))

  def forward(self, *img: torch.Tensor) -> Tuple[torch.Tensor]:
    rgb = self.dist.sample()
    return ((im * rgb[None, :, None, None]) / rgb.sum() for im in img)

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
    
    color_shift=ColorShift()
    color_shift.setup(device=device)

    criterion = nn.BCELoss()

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
            fake_output=guide_filter(fake, fake_out, r=1)

            fake_blur=guild_filter(fake_output, fake_output, r=5)#Part 1. Blur GAN
            real_blur=guild_filter(real, real, r=5)

            fake_gray, real_gray=color_shift(fake_output, real)#Part 2.Gray GAN

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
            fake_output=guide_filter(fake, fake_out, r=1)
            fake_blur=guild_filter(fake_output, fake_output, r=5)
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




