import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

def denormalize(im, mean=0.5, std=0.5):
  return im * std + mean

class VGG19(nn.Module):
    def __init__(self, output_index=26):
        super().__init__()
        self.vgg=models.vgg19(pretrained=True)
        self.features=self.vgg.features
        self.output_index=output_index
    def process(self, x):
        normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        x=normalize(denormalize(x))
    def forward(self, x):
        x=self.features[:self.output_index](x)
        return x

if __name__=="__main__":
    x=torch.rand((1, 3, 224, 224), dtype=float32)
    net=VGG19()
    x=net(x)
