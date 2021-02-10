import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import *

class generator(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.down1=ConvLRelu(n_channels, 32, kernel_size=7, stride=1,  padding=3)
        self.down2_1=ConvLRelu(32, 32, kernel_size=3, stride=2)
        self.down2_2=ConvLRelu(32, 64, kernel_size=3, stride=1)
        self.down3_1=ConvLRelu(64, 64, kernel_size=3, stride=2)
        self.down3_2=ConvLRelu(64, 128, kernel_size=3, stride=1)

        #self.block=ResidualBlock(128, 128, kernel_size=3, stride=1)
        self.block = nn.Sequential(*[ResidualBlock(128, 128, kernel_size=3, stride=1) for i in range(4)])
        self.resize=nn.UpsamplingBilinear2d(scale_factor=2)

        self.up1=ConvLRelu(128, 64, kernel_size=3, stride=1)
        self.up2=ConvLRelu(64, 64, kernel_size=3, stride=1)
        self.up3=ConvLRelu(64, 32, kernel_size=3, stride=1)
        self.up4=ConvLRelu(32, 32, kernel_size=3, stride=1)
        self.out=nn.Conv2d(32, n_classes, kernel_size=7, stride=1,  padding=3)
        self.act = nn.Tanh()
    def forward(self,x):
        x1=self.down1(x)
        x2=self.down2_1(x1)
        x2=self.down2_2(x2)
        x3=self.down3_1(x2)
        x3=self.down3_2(x3)
        x3=self.block(x3)
        x3=self.up1(x3)
        x4=self.resize(x3)
        x4=self.up2(x2+x4)
        x4=self.up3(x4)
        x5=self.resize(x4)
        x5=self.up4(x1+x5)
        x5=self.out(x5)
        return self.act(x5)

        
class discriminator(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.func1_1=ConvSpecLRelu(n_channels, 32, kernel_size=3, stride=2)
        self.func1_2=ConvSpecLRelu(32, 32, kernel_size=3, stride=1)

        self.func2_1=ConvSpecLRelu(32, 64, kernel_size=3, stride=2)
        self.func2_2=ConvSpecLRelu(64, 64, kernel_size=3, stride=1)

        self.func3_1=ConvSpecLRelu(64, 128, kernel_size=3, stride=2)
        self.func3_2=ConvSpecLRelu(128, 128, kernel_size=3, stride=1)

        self.out=nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
    
    def forward(self,x):
        x=self.func1_1(x)
        x=self.func1_2(x)
        x=self.func2_1(x)
        x=self.func2_2(x)
        x=self.func3_1(x)
        x=self.func3_2(x)
        return self.out(x)
