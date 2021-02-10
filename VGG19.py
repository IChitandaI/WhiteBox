import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
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


class VGGCaffePreTrained(nn.Module):
  cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,
         'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

  def __init__(self, weights_path: str = './checkpoints/vgg19_no_fc.npy', output_index: int = 26) -> None:
    super().__init__()
    try:
      data_dict: dict = np.load(weights_path, encoding='latin1', allow_pickle=True).item()
      self.features = self.make_layers(self.cfg, data_dict)
      del data_dict
    except FileNotFoundError as e:
      print(ERROR, "weights_path:", weights_path,
            'does not exits!, if you want to training must download pretrained weights')
    self.output_index = output_index

  def _process(self, x):
    # NOTE 图像范围为[-1~1]，先denormalize到0-1再归一化
    rgb = denormalize(x) * 255  # to 255
    bgr = rgb[:, [2, 1, 0], :, :]  # rgb to bgr
    return self.vgg_normalize(bgr)  # vgg norm

  def setup(self, device: torch.device):
    mean: torch.Tensor = torch.tensor([103.939, 116.779, 123.68], device=device)
    mean = mean[None, :, None, None]
    self.vgg_normalize = lambda x: x - mean

  def _forward_impl(self, x):
    x = self._process(x)
    # NOTE get output with out relu activation
    x = self.features[:self.output_index](x)
    return x

  def forward(self, x):
    return self._forward_impl(x)

  @staticmethod
  def get_conv_filter(data_dict, name):
    return data_dict[name][0]

  @staticmethod
  def get_bias(data_dict, name):
    return data_dict[name][1]

  @staticmethod
  def get_fc_weight(data_dict, name):
    return data_dict[name][0]

  def make_layers(self, cfg, data_dict, batch_norm=False) -> nn.Sequential:
    layers = []
    in_channels = 3
    block = 1
    number = 1
    for v in cfg:
      if v == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        block += 1
        number = 1
      else:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        with torch.no_grad():
          """ set value """
          weight = torch.FloatTensor(self.get_conv_filter(data_dict, f'conv{block}_{number}'))
          weight = weight.permute((3, 2, 0, 1))
          bias = torch.FloatTensor(self.get_bias(data_dict, f'conv{block}_{number}'))
          conv2d.weight.set_(weight)
          conv2d.bias.set_(bias)
        number += 1
        if batch_norm:
          layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
          layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    return nn.Sequential(*layers)
