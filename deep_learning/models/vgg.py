import torch
import torch.nn as nn

from deep_learning.models.se_module import SELayer

__all__ = [
  'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
  'se_vgg11', 'se_vgg11_bn', 'se_vgg13', 'se_vgg13_bn', 'se_vgg16', 'se_vgg16_bn',
  'se_vgg19', 'se_vgg19_bn']


class VGG(nn.Module):
  def __init__(self, features, num_classes=1000, init_weights=True):
    super(VGG, self).__init__()
    self.features = features
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    self.classifier = nn.Sequential(
      nn.Linear(512 * 7 * 7, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(4096, num_classes),
    )
    if init_weights:
      self._initialize_weights()

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, use_seblock=False):
  layers = []
  in_channels = 3
  out_channels = [64, 128, 256, 512, 512]
  for i, v in enumerate(cfg):
    for _ in range(v):
      conv2d = nn.Conv2d(in_channels, out_channels[i], kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(out_channels[i]), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = out_channels[i]
    if use_seblock:
      layers += [SELayer(in_channels, reduction=16)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
  return nn.Sequential(*layers)


cfgs = {
  'A': [1, 1, 2, 2, 2],
  'B': [2, 2, 2, 2, 2],
  'D': [2, 2, 3, 3, 3],
  'E': [2, 2, 4, 4, 4],
}


def vgg11(**kwargs):
  return VGG(make_layers(cfgs['A']), **kwargs)


def vgg11_bn(**kwargs):
  return VGG(make_layers(cfgs['A'], batch_norm=True), **kwargs)


def vgg13(**kwargs):
  return VGG(make_layers(cfgs['B']), **kwargs)


def vgg13_bn(**kwargs):
  return VGG(make_layers(cfgs['B'], batch_norm=True), **kwargs)


def vgg16(**kwargs):
  return VGG(make_layers(cfgs['D']), **kwargs)


def vgg16_bn(**kwargs):
  return VGG(make_layers(cfgs['D'], batch_norm=True), **kwargs)


def vgg19(**kwargs):
  return VGG(make_layers(cfgs['E']), **kwargs)


def vgg19_bn(**kwargs):
  return VGG(make_layers(cfgs['E'], batch_norm=True), **kwargs)


def se_vgg11(**kwargs):
  return VGG(make_layers(cfgs['A'], use_seblock=True), 
             **kwargs)


def se_vgg11_bn(**kwargs):
  return VGG(make_layers(cfgs['A'], batch_norm=True, use_seblock=True),
             **kwargs)


def se_vgg13(**kwargs):
  return VGG(make_layers(cfgs['B'], use_seblock=True), 
             **kwargs)


def se_vgg13_bn(**kwargs):
  return VGG(make_layers(cfgs['B'], batch_norm=True, use_seblock=True), 
             **kwargs)


def se_vgg16(**kwargs):
  return VGG(make_layers(cfgs['D'], use_seblock=True), 
             **kwargs)


def se_vgg16_bn(**kwargs):
  return VGG(make_layers(cfgs['D'], batch_norm=True, use_seblock=True), 
             **kwargs)


def se_vgg19(**kwargs):
  return VGG(make_layers(cfgs['E'], use_seblock=True), 
             **kwargs)


def se_vgg19_bn(**kwargs):
  return VGG(make_layers(cfgs['E'], batch_norm=True, use_seblock=True), 
             **kwargs)
