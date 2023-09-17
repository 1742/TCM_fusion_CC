import sys

import torch
import torchvision.models
from torch import nn
from torch.nn import init
from model.ResNet.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck
from torchvision import models

import os
import json
from typing import Type, Optional, Callable, Union


class SE_ResNet(nn.Module):
    def __init__(
            self,
            block: [BasicBlock, Bottleneck],
            layers: list[int],
            in_channels: int = 3,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[list[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            include_top: bool = True
    ):
        super(SE_ResNet, self).__init__()
        self.fc_cells = 512 * block.expansion
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.se1 = SE_Block(64 * block.expansion)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.se2 = SE_Block(128 * block.expansion)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.se3 = SE_Block(256 * block.expansion)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.se4 = SE_Block(512 * block.expansion)

        self.include_top = include_top
        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.zero_init_residual = zero_init_residual
        self._init_weights()

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.layer4(x)
        x = self.se4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    def get_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.layer4(x)
        x = self.se4(x)

        return x

    def get_embeddings(self, x: torch.Tensor):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.layer4(x)
        x = self.se4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


class SE_Block(nn.Module):
    def __init__(self, in_channels: int, r: int = 16):
        super(SE_Block, self).__init__()
        self.in_channels = in_channels

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.in_channels, int(self.in_channels / r), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(self.in_channels / r), self.in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se_block(x)


def se_resnet18(in_channels=3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet18-f37072fd.pth
    return SE_ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, include_top=include_top)


def se_resnet34(in_channels=3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return SE_ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes, include_top=include_top)


def se_resnet50(in_channels=3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return SE_ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes, include_top=include_top)


def se_resnet101(in_channels=3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return SE_ResNet(Bottleneck, [3, 4, 23, 3], in_channels=in_channels, num_classes=num_classes, include_top=include_top)


def se_resnext50_32x4d(in_channels=3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return SE_ResNet(Bottleneck, [3, 4, 6, 3],
                  in_channels=in_channels,
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def se_resnext101_32x8d(in_channels=3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return SE_ResNet(Bottleneck, [3, 4, 23, 3],
                  in_channels=in_channels,
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    se_resnet = se_resnet34(num_classes=2).to(device)
    print(se_resnet)

    x = torch.randint(255, (1, 3, 448, 448)).float().to(device)
    out = se_resnet(x / 255.)
    print(out, out.size())


