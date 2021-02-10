from network.net_build import *
from superpix import slic, adaptive_slic, sscolor
from VGG19 import VGG19
from predata import Data_set
from guild_filter_code import guide_filter

import logging
import os
import sys
from glob import glob
import itertools

import torch.nn as nn
from torch import optim
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.distributions import Distribution
from functools import partial
from joblib import Parallel, delayed
from typing import List, Tuple

dir_img = './data/fake/'
dir_real = './data/real/'
dir_checkpoint = './checkpoints/'
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

class VariationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        b, c, h, w=x.shape
        tv_h=torch.mean((x[:, :, 1:, :]-x[:, :, :-1, :])**2)
        tv_w=torch.mean((x[:, :, :, 1:]-x[:, :, :, :-1])**2)
        return (tv_h+tv_w)/(h*w*3)# try h*w*c?


def pretrain(
          G_net,
          device,
          epochs=100,
          batch_size=5,
          lr=2e-4,
          val_percent=0.1,
          save_cp=True):

    train_data=Data_set(dir_img, dir_real)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    batches=len(train_loader)
    G=G_net
    G_optimizer=torch.optim.Adam(G.parameters(), lr=lr)
    l1_loss=nn.L1Loss('mean')

    for epoch in range(epochs):
        G.train()
        num=0
        for batch in train_loader:
            fake=batch['fake'].to(device)
            num+=1
            fake_out=G(fake)
            loss_sum=l1_loss(fake,fake_out)
            G_optimizer.zero_grad()
            loss_sum.backward()
            G_optimizer.step()
            if(num % 1 == 0):
                print("epoch:{}/{} batch:{}/{}".format(epoch,epochs,num,batches))
    torch.save(G.state_dict(), dir_checkpoint +f'checkpoint_epoch{epochs}.pth')

if __name__ == "__main__":
    G_net=generator(n_channels=3, n_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    G_net.to(device=device)
    pretrain(G_net, device=device)


