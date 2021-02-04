import torch
import torch.nn as nn
import torchvision.models as models

class VGG19(nn.Module):
    def __init__(self, output_index=26):
        self.vgg=models.vgg19(pretrained=True)
        self.features=self.vgg.features

    #def process(self, x):

    def forward(self, x):
        x=self.process(x)
        x=self.features[:self.output_index](x)
        return x

if __name__=="__main__":
    x=torch.rand((64, 64), dtype=float32)
    net=VGG19()
    x=net(x)
