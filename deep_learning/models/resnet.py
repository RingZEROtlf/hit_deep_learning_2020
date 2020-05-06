import torch
import torch.nn as nn

from deep_learning.models.se_module import SELayer

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
           'se_resnet18', 'se_resnet34', 'se_resnet50', 'se_resnet101', 
           'se_resnet152']


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=dilation, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None,
               use_seblock=False):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.se = SELayer(planes, reduction=16) if use_seblock else None
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.se is not None:
      out = self.se(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, 
               use_seblock=False):
    super(Bottleneck, self).__init__()
    self.conv1 = conv1x1(inplanes, planes)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes, stride)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = conv1x1(planes, planes * self.expansion)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.se = SELayer(planes * self.expansion, reduction=16) if use_seblock else None
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.se is not None:
      out = self.se(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000, use_seblock=False):
    super(ResNet, self).__init__()

    self.inplanes = 64

    self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0], use_seblock=use_seblock)
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                   use_seblock=use_seblock)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                   use_seblock=use_seblock)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                   use_seblock=use_seblock)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, block, planes, blocks, stride=1, use_seblock=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        conv1x1(self.inplanes, planes * block.expansion, stride),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, use_seblock=use_seblock))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, use_seblock=use_seblock))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x


def resnet18(**kwargs):
  return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
  return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
  return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
  return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
  return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def se_resnet18(**kwargs):
  return ResNet(BasicBlock, [2, 2, 2, 2], use_seblock=True, **kwargs)


def se_resnet34(**kwargs):
  return ResNet(BasicBlock, [3, 4, 6, 3], use_seblock=True, **kwargs)


def se_resnet50(**kwargs):
  return ResNet(Bottleneck, [3, 4, 6, 3], use_seblock=True, **kwargs)


def se_resnet101(**kwargs):
  return ResNet(Bottleneck, [3, 4, 23, 3], use_seblock=True, **kwargs)


def se_resnet152(**kwargs):
  return ResNet(Bottleneck, [3, 8, 36, 3], use_seblock=True, **kwargs)
