import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.convlrelu=nn.sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.LeakyReLU(replace=True)
        )
    def forward(x):
        return self.convlrelu(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.func=nn.sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.LeakyReLU(replace=True)
        )
        self.conv=nn.Conv2d(out_channels, out_channels, kernel_size, stride)
    def forward(x):
        x1=self.func(x)
        x1=self.conv(x1)
        return x+x1

class ConvSpecLRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.func=nn.sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride)),
            nn.LeakyReLU(replace=True)
        )