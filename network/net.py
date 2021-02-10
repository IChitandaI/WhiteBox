import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ConvLRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.convlrelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.convlrelu(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding=1):
        super().__init__()
        self.func=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.LeakyReLU(inplace=True)
        )
        self.conv=nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=padding)
    def forward(self,x):
        x1=self.func(x)
        x1=self.conv(x1)
        return x+x1

class ConvSpecLRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding=1):
        super().__init__()
        self.func=nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding=padding)),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        x=self.func(x)
        return x