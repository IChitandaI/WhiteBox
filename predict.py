from network.net_build import *
from predata import Data_set
from data_vis import plot_img_and_mask
from guide_filter_code import guide_filter

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
from torchvision import transforms


net_path='./checkpoints/checkpoint_epoch100.pth'
predict_pic_path='E:/WhiteBox/data/fake/2013-11-10 07_43_23.jpg'

def pre_dict(net,
             full_img,
             device):
    net.eval()
    img=torch.from_numpy(Data_set.resize(full_img)).type(torch.FloatTensor)
    img=img.unsqueeze(0)
    img=img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output=net(img)
        output=guide_filter(output, output, r=5, eps=2e-1)
        output=output.squeeze(0)
        tf=transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize(full_img.size[1]),
            transforms.ToTensor()]
        )
        output=tf(output.cpu())
    return output.cpu().numpy()

def out_to_image(out):
    return Image.fromarray(np.asarray(out*255).astype(np.uint8).transpose((1,2,0)))

if __name__=="__main__":
    net = generator(n_channels=3, n_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(net_path, map_location=device))

    img = Image.open(predict_pic_path)
    out=pre_dict(net,img,device)
    print(out.shape)
    out_image=out_to_image(out)
    plot_img_and_mask(out_image)

