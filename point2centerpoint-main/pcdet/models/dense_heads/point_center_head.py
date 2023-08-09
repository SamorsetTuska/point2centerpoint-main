import copy
from collections import OrderedDict
import time
from math import ceil, floor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.init import kaiming_normal_
from ...utils import loss_utils
from .center_loss import SetCriterion, HungarianMatcher, HungarianMatcherDynamicK
from .center_loss_align import SetCriterion_align
from ..model_utils import centernet_utils
from scipy.ndimage.filters import gaussian_filter
from ..model_utils import model_nms_utils
import torch.nn.functional as F


class Heatmap(nn.Module):
    def __init__(self, num_classes, num_features_in=128, feature_size=64, init_bias=-2.19):
        super(Heatmap, self).__init__()

        self.conv1 = nn.Conv2d(2 * num_features_in, feature_size, kernel_size=3, padding=1, bias=True)
        # self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_classes, kernel_size=3, padding=1)
        # self.output = nn.Conv2d(num_features_in, num_classes, kernel_size=1, padding=0)
        nn.init.constant_(self.output.bias, init_bias)

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.output(out)
        # out = self.output(x)

        return out.contiguous()


class Densitymap(nn.Module):
    def __init__(self, num_features_in=128, feature_size=64, init_bias=-2.19):
        super(Densitymap, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1, bias=True)
        # self.bn1 = nn.BatchNorm2d(feature_size)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(feature_size)

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1, bias=True)
        # self.conv3.bias.data.fill_(init_bias)
        # self.bn3 = nn.BatchNorm2d(feature_size)

        self.output = nn.Conv2d(feature_size, 3, kernel_size=3, padding=1)
        self.upsampled = nn.Upsample(scale_factor=8, mode='bilinear')

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        # out = self.bn3(out)
        out = self.act(out)

        out = self.output(out)
        # out = self.upsampled(out)

        return out.contiguous()


class Mask(nn.Module):
    def __init__(self, num_features_in=64):
        super(Mask, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, 32, kernel_size=3, padding=1, bias=True)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 4, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out.contiguous()


class SharedConvResV1(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(SharedConvResV1, self).__init__()
        # 残差模块降采样
        self.conv1 = nn.Conv2d(num_features_in, 2 * num_features_in, kernel_size=3, stride=2, padding=1, bias=False)
        self.downsample1 = nn.Conv2d(num_features_in, 2 * num_features_in, kernel_size=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * num_features_in)
        self.bn2 = nn.BatchNorm2d(2 * num_features_in)
        self.act = nn.ReLU()

        self.conv2 = nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2 * num_features_in)

        # 增加一个卷积
        ## TODO: 记得恢复注释
        self.conv6 = nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(2 * num_features_in)

        # 残差模块降采样
        self.conv3 = nn.Conv2d(2 * num_features_in, 4 * num_features_in, kernel_size=3, stride=2, padding=1, bias=False)
        self.downsample2 = nn.Conv2d(2 * num_features_in, 4 * num_features_in, kernel_size=1, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(4 * num_features_in)
        self.bn5 = nn.BatchNorm2d(4 * num_features_in)

        self.conv4 = nn.Conv2d(4 * num_features_in, 4 * num_features_in, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(4 * num_features_in)
        ## TODO: 记得恢复注释
        self.conv7 = nn.Conv2d(4 * num_features_in, 4 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(4 * num_features_in)

        # 1个k=1的卷积核，将输入的通道数变为1/8
        self.conv5 = nn.Conv2d(4 * num_features_in, feature_size, kernel_size=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(feature_size)

        self.transconv = nn.ConvTranspose2d(4 * num_features_in, 2 * num_features_in, kernel_size=2, stride=2,
                                            bias=False)
        self.bn8 = nn.BatchNorm2d(2 * num_features_in)
        self.down = nn.Sequential(
            nn.Conv2d(4 * num_features_in, feature_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU())

    def forward(self, x, training=True):
        # 残差模块降采样
        indentity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        indentity = self.downsample1(indentity)
        indentity = self.bn3(indentity)
        out += indentity
        # out_ = out
        out = self.act(out)
        ## TODO: 记得恢复注释
        out = self.conv6(out)
        out = self.bn9(out)
        out = self.act(out)
        out_ = out
        # 残差模块降采样
        indentity = out
        out = self.conv3(out)
        out = self.bn4(out)
        out = self.act(out)
        out = self.conv4(out)
        out = self.bn5(out)
        indentity = self.downsample2(indentity)
        indentity = self.bn6(indentity)
        out += indentity
        out = self.act(out)

        ## TODO: 记得恢复注释
        out = self.conv7(out)
        out = self.bn10(out)
        out = self.act(out)
        if training:
            out_up = self.transconv(out)
            out_up = self.bn8(out_up)
            out_ = torch.cat((out_, out_up), dim=1)
            out_ = self.act(out_)
            out_ = self.down(out_)
        # 用1个k=1的卷积核把通道数降到1/8
        out = self.conv5(out)
        out = self.bn7(out)
        out = self.act(out)

        # out = out + out_
        return x, out_, out


class SharedConvResV2(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(SharedConvResV2, self).__init__()
        # 残差模块降采样
        self.conv1 = nn.Conv2d(num_features_in, 2 * num_features_in, kernel_size=3, stride=2, padding=1, bias=False)
        self.downsample1 = nn.Conv2d(num_features_in, 2 * num_features_in, kernel_size=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * num_features_in)
        self.bn2 = nn.BatchNorm2d(2 * num_features_in)
        self.act = nn.ReLU()

        self.conv2 = nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2 * num_features_in)

        # 增加一个卷积
        ## TODO: 记得恢复注释
        self.conv6 = nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(2 * num_features_in)

        self.adp_pool = nn.AdaptiveAvgPool2d((75, 66))

        # 残差模块降采样
        self.conv3 = nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(2 * num_features_in)
        self.conv4 = nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(2 * num_features_in)
        ## TODO: 记得恢复注释
        self.conv7 = nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(2 * num_features_in)

        # 1个k=1的卷积核，将输入的通道数变为1/8
        self.conv5 = nn.Conv2d(2 * num_features_in, feature_size, kernel_size=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(feature_size)

        self.up = nn.Upsample(scale_factor=4 / 3, mode='bilinear', align_corners=True)
        self.down = nn.Sequential(nn.AdaptiveAvgPool2d((75, 66)),
                                  nn.Conv2d(4 * num_features_in, feature_size, kernel_size=1, stride=1, padding=0,
                                            bias=False),
                                  nn.BatchNorm2d(feature_size),
                                  nn.ReLU())

    def forward(self, x, training=True):
        # 残差模块降采样
        indentity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        indentity = self.downsample1(indentity)
        indentity = self.bn3(indentity)
        out += indentity
        out = self.act(out)
        ## TODO: 记得恢复注释
        out = self.conv6(out)
        out = self.bn9(out)
        out = self.act(out)
        out_ = out
        out = self.adp_pool(out)
        # 残差模块降采样
        indentity = out
        out = self.conv3(out)
        out = self.bn5(out)
        out = self.act(out)
        out = self.conv4(out)
        out = self.bn6(out)
        out += indentity
        out = self.act(out)

        ## TODO: 记得恢复注释
        out = self.conv7(out)
        out = self.bn10(out)
        out = self.act(out)
        if training:
            out_up = self.up(out)
            out_ = torch.cat((out_, out_up), dim=1)
            out_ = self.act(out_)
            out_ = self.down(out_)
        # 用1个k=1的卷积核把通道数降到1/8
        out = self.conv5(out)
        out = self.bn7(out)
        out = self.act(out)
        return x, out_, out


class SharedConvResV3(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(SharedConvResV3, self).__init__()
        # 残差模块降采样
        self.act = nn.ReLU()
        self.block1 = nn.Sequential(
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in))
        self.conv1 = nn.Sequential(nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features_in),
                                   nn.ReLU())

        self.block2 = nn.Sequential(
            nn.Conv2d(num_features_in, 2 * num_features_in, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
            nn.ReLU(),
            nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2 * num_features_in))

        self.down_side = nn.Sequential(
            nn.Conv2d(num_features_in, 2 * num_features_in, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(2 * num_features_in))

        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
            nn.ReLU())

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Sequential(
            nn.Conv2d(3 * num_features_in, feature_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, feature_size, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU())

    def forward(self, x, training=True):
        # 残差模块
        indentity = x
        out = self.block1(x)
        out += indentity
        out = self.act(out)

        out = self.conv1(out)
        out_ = out

        # 残差模块降采样
        indentity = out
        out = self.block2(out)
        indentity = self.down_side(indentity)
        out += indentity
        out = self.act(out)
        out = self.conv2(out)

        if training:
            out_up = self.up(out)
            out_ = torch.cat((out_, out_up), dim=1)
            out_ = self.act(out_)
            out_ = self.down(out_)
        # 用1个k=1的卷积核把通道数降到1/8
        out = self.conv3(out)
        return x, out_, out


class SharedConvResV4(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(SharedConvResV4, self).__init__()
        # 残差模块降采样
        self.conv1 = nn.Conv2d(num_features_in, 2 * num_features_in, kernel_size=3, stride=2, padding=1, bias=False)
        self.downsample1 = nn.Conv2d(num_features_in, 2 * num_features_in, kernel_size=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * num_features_in)
        self.bn2 = nn.BatchNorm2d(2 * num_features_in)
        self.act = nn.ReLU()

        self.conv2 = nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2 * num_features_in)

        # 增加一个卷积
        ## TODO: 记得恢复注释
        self.conv6 = nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(2 * num_features_in)

        # 残差模块降采样
        self.conv3 = nn.Conv2d(2 * num_features_in, 4 * num_features_in, kernel_size=3, stride=2, padding=1, bias=False)
        self.downsample2 = nn.Conv2d(2 * num_features_in, 4 * num_features_in, kernel_size=1, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(4 * num_features_in)
        self.bn5 = nn.BatchNorm2d(4 * num_features_in)

        self.conv4 = nn.Conv2d(4 * num_features_in, 4 * num_features_in, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(4 * num_features_in)
        ## TODO: 记得恢复注释
        self.conv7 = nn.Conv2d(4 * num_features_in, 4 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(4 * num_features_in)

        # 1个k=1的卷积核，将输入的通道数变为1/8
        self.conv5 = nn.Conv2d(4 * num_features_in, feature_size, kernel_size=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(feature_size)

        self.up1 = nn.Sequential(
            nn.Conv2d(4 * num_features_in, 2 * num_features_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.up2 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, num_features_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.down1 = nn.Sequential(
            nn.Conv2d(num_features_in, 2 * num_features_in, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * num_features_in))

        self.down2 = nn.Sequential(
            # nn.Conv2d(4 * num_features_in, feature_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(2 * num_features_in, feature_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_size))
        self.conv8 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
            nn.ReLU())
        self.conv9 = nn.Sequential(
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU())
        self.conv10 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
            nn.ReLU())

        # self.self_attn = nn.MultiheadAttention(64, 8, dropout=0.0, batch_first=True)

        if nn.Conv2d in self._modules:
            self._modules[nn.Conv2d].apply(self.init_weights)
        if nn.BatchNorm2d in self._modules:
            self._modules[nn.BatchNorm2d].apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x, training=True):
        # 残差模块降采样
        indentity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        indentity = self.downsample1(indentity)
        indentity = self.bn3(indentity)
        out += indentity
        # out_ = out
        out = self.act(out)
        out = self.conv6(out)
        out = self.bn9(out)
        out = self.act(out)
        out_ = out
        # 残差模块降采样
        indentity = out
        out = self.conv3(out)
        out = self.bn4(out)
        out = self.act(out)
        out = self.conv4(out)
        out = self.bn5(out)
        indentity = self.downsample2(indentity)
        indentity = self.bn6(indentity)
        out += indentity
        out = self.act(out)

        out = self.conv7(out)
        out = self.bn10(out)
        out = self.act(out)
        if training:
            out_ = out_ + self.up1(out)
            # out_ = self.act(out_)
            out_ = self.conv8(out_)

            x = x + self.up2(out_)
            # x = self.act(x)
            x = self.conv9(x)

            out_ = out_ + self.down1(x)
            # out_ = self.act(out_)
            out_ = self.conv10(out_)
            out_ = self.down2(out_)
        # 用1个k=1的卷积核把通道数降到1/8
        out = self.conv5(out)
        out = self.bn7(out)
        # out = self.act(out)

        out = out + out_
        out_ = self.act(out_)
        out = self.act(out)
        # shape = out.shape
        # out = out.view(out.shape[0], out.shape[1], out.shape[2] * out.shape[3]).permute(0, 2, 1)
        #
        # out = self.self_attn(out, out, out)[0]
        # out = out.permute(0, 2, 1).view(shape)
        # # out = self.conv8(out)

        return x, out_, out


class ChannelAttention(nn.Module):
    def __init__(self, num_features_in, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(num_features_in, num_features_in // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features_in // ratio, num_features_in, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class SharedConvResV5(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(SharedConvResV5, self).__init__()
        self.act = nn.ReLU()
        # self.act = nn.LeakyReLU(0.2, inplace=True)
        # 残差模块降采样
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU()
            # nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool1 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True)
        )

        # self.downsample1 = nn.Sequential(
        #     nn.Conv2d(num_features_in, 2 * num_features_in, kernel_size=1, stride=2, bias=False),
        #     nn.BatchNorm2d(2 * num_features_in))

        # 增加一个卷积
        ## TODO: 记得恢复注释
        self.conv6 = nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(2 * num_features_in)

        # 残差模块降采样
        self.conv3 = nn.Conv2d(2 * num_features_in, 4 * num_features_in, kernel_size=3, stride=2, padding=1, bias=False)
        self.downsample2 = nn.Conv2d(2 * num_features_in, 4 * num_features_in, kernel_size=1, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(4 * num_features_in)
        self.bn5 = nn.BatchNorm2d(4 * num_features_in)

        self.conv4 = nn.Conv2d(4 * num_features_in, 4 * num_features_in, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(4 * num_features_in)
        ## TODO: 记得恢复注释
        self.conv7 = nn.Conv2d(4 * num_features_in, 4 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(4 * num_features_in)

        # 1个k=1的卷积核，将输入的通道数变为1/8
        self.conv5 = nn.Conv2d(4 * num_features_in, feature_size, kernel_size=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(feature_size)

        self.transconv = nn.ConvTranspose2d(4 * num_features_in, 2 * num_features_in, kernel_size=2, stride=2,
                                            bias=False)
        self.bn8 = nn.BatchNorm2d(2 * num_features_in)
        self.down = nn.Sequential(
            nn.Conv2d(4 * num_features_in, feature_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        )

        # self.channel_attention = ChannelAttention(feature_size)
        # self.spatial_attention = SpatialAttention()

        if nn.Conv2d in self._modules:
            self._modules[nn.Conv2d].apply(self.init_weights)
        if nn.BatchNorm2d in self._modules:
            self._modules[nn.BatchNorm2d].apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x, training=True):
        # 残差模块降采样
        out = torch.cat([self.conv1(x), self.pool1(x)], dim=1)
        # out = out + self.downsample1(x)
        out = self.act(out)
        ## TODO: 记得恢复注释
        out = self.conv6(out)
        out = self.bn9(out)
        out = self.act(out)
        out_ = out
        # 残差模块降采样
        indentity = out
        out = self.conv3(out)
        out = self.bn4(out)
        out = self.act(out)
        out = self.conv4(out)
        out = self.bn5(out)
        indentity = self.downsample2(indentity)
        indentity = self.bn6(indentity)
        out += indentity
        out = self.act(out)

        ## TODO: 记得恢复注释
        out = self.conv7(out)
        out = self.bn10(out)
        out = self.act(out)
        if training:
            out_ = torch.cat((out_, self.bn8(self.transconv(out))), dim=1)
            out_ = self.act(out_)
            out_ = self.down(out_)
        # 用1个k=1的卷积核把通道数降到1/8
        out = self.conv5(out)
        out = self.bn7(out)
        out = self.act(out)

        out = out + out_
        # out = out * self.channel_attention(out)
        # out = out * self.spatial_attention(out)
        return x, out_, out


class SharedConvResV6(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(SharedConvResV6, self).__init__()
        self.act = nn.ReLU()
        self.conv0 = nn.Sequential(
            nn.Conv2d(num_features_in, num_features_in // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, num_features_in // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU()
        )
        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
        )

        # 增加一个卷积
        ## TODO: 记得恢复注释
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
            nn.ReLU()
        )

        # 残差模块降采样
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
            nn.Conv2d(num_features_in, 4 * num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4 * num_features_in),
        )
        self.side_conv1 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, 4 * num_features_in, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(4 * num_features_in)
        )

        ## TODO: 记得恢复注释
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * num_features_in, 4 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4 * num_features_in),
            nn.ReLU()
        )

        # 1个k=1的卷积核，将输入的通道数变为1/8
        self.conv5 = nn.Sequential(
            nn.Conv2d(4 * num_features_in, feature_size, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        )

        self.transconv1 = nn.Sequential(
            nn.ConvTranspose2d(4 * num_features_in, 2 * num_features_in, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
            nn.ReLU())
        self.transconv2 = nn.Sequential(
            nn.ConvTranspose2d(4 * num_features_in, num_features_in, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU())
        self.down = nn.Sequential(
            nn.Conv2d(4 * num_features_in, feature_size, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        )

        if nn.Conv2d in self._modules:
            self._modules[nn.Conv2d].apply(self.init_weights)
        if nn.BatchNorm2d in self._modules:
            self._modules[nn.BatchNorm2d].apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x, training=True):
        # out = self.act(self.conv0(x) + x)
        # out = self.act(torch.cat([self.conv1(out), self.pool1(out)], dim=1))
        x = self.act(self.conv0(x) + x)
        out = self.act(torch.cat([self.conv1(x), self.pool1(x)], dim=1))
        out = self.conv2(out)
        out_ = out
        out = self.conv3(out) + self.side_conv1(out)
        out = self.act(out)
        out = self.conv4(out)
        out_ = torch.cat((out_, self.transconv1(out)), dim=1)
        if training:
            # out_ = torch.cat((out_, self.transconv1(out)), dim=1)
            x = torch.cat((x, self.transconv2(out_)), dim=1)
        out_ = self.down(out_)
        out = self.conv5(out)
        # out = torch.cat((out, out_), dim=1)
        out = out + out_
        return x, out_, out


class SharedConvResV7(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(SharedConvResV7, self).__init__()
        self.act = nn.ReLU()
        self.conv0 = nn.Sequential(
            nn.Conv2d(num_features_in, num_features_in // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, num_features_in // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in),
        )
        self.conv0_ = nn.Sequential(
            nn.Conv2d(num_features_in, num_features_in // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, num_features_in // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU()
        )
        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
        )

        # 增加一个卷积
        ## TODO: 记得恢复注释
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(2 * num_features_in),
        #     nn.ReLU()
        # )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, num_features_in // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, num_features_in // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, 2 * num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, num_features_in // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, num_features_in // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, 2 * num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
            nn.Conv2d(num_features_in, 4 * num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4 * num_features_in),
        )

        self.side_conv1 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, 4 * num_features_in, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(4 * num_features_in)
        )

        ## TODO: 记得恢复注释
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * num_features_in, 4 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4 * num_features_in),
            nn.ReLU()
        )

        # 1个k=1的卷积核，将输入的通道数变为1/8
        self.conv5 = nn.Sequential(
            nn.Conv2d(4 * num_features_in, feature_size, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        )

        self.transconv = nn.ConvTranspose2d(4 * num_features_in, 2 * num_features_in, kernel_size=2, stride=2,
                                            bias=False)
        self.bn8 = nn.BatchNorm2d(2 * num_features_in)
        self.transconv2 = nn.Sequential(
            nn.ConvTranspose2d(4 * num_features_in, num_features_in, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU())
        self.down = nn.Sequential(
            nn.Conv2d(4 * num_features_in, feature_size, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        )

        if nn.Conv2d in self._modules:
            self._modules[nn.Conv2d].apply(self.init_weights)
        if nn.BatchNorm2d in self._modules:
            self._modules[nn.BatchNorm2d].apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x, training=True):
        # 残差模块降采样
        out = self.act(self.conv0(x) + x)
        out = self.act(self.conv0_(out) + out)
        out = self.act(torch.cat([self.conv1(out), self.pool1(out)], dim=1))
        # out = self.conv2(out)
        out_ = out
        # 残差模块降采样
        out = self.act(self.conv2_1(out) + out)
        out = self.act(self.conv2_2(out) + out)
        out = self.conv3(out) + self.side_conv1(out)
        out = self.act(out)
        out = self.conv4(out)
        out_ = torch.cat((out_, self.bn8(self.transconv(out))), dim=1)
        if training:
            x = torch.cat((x, self.transconv2(out_)), dim=1)
        out_ = self.act(out_)
        out_ = self.down(out_)
        # 用1个k=1的卷积核把通道数降到1/8
        out = self.conv5(out)

        out = out + out_
        return x, out_, out


class SharedConvResV8(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(SharedConvResV8, self).__init__()
        self.act = nn.ReLU()
        self.conv0 = nn.Sequential(
            nn.Conv2d(num_features_in, num_features_in // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, num_features_in // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(num_features_in, 2 * num_features_in, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, num_features_in // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, num_features_in // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in // 2),
            nn.ReLU(),
            nn.Conv2d(num_features_in // 2, 2 * num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
        )

        # 增加一个卷积
        ## TODO: 记得恢复注释
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
            nn.ReLU(),
            nn.Conv2d(2 * num_features_in, 2 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
            nn.ReLU(True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(2 * num_features_in, 4 * num_features_in, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * num_features_in),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4 * num_features_in, num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
            nn.Conv2d(num_features_in, num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU(),
            nn.Conv2d(num_features_in, 4 * num_features_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4 * num_features_in),
        )

        ## TODO: 记得恢复注释
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * num_features_in, 4 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4 * num_features_in),
            nn.ReLU(),
            nn.Conv2d(4 * num_features_in, 4 * num_features_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4 * num_features_in),
            nn.ReLU(True),
        )

        # 1个k=1的卷积核，将输入的通道数变为1/8
        self.conv5 = nn.Sequential(
            nn.Conv2d(4 * num_features_in, feature_size, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(4 * num_features_in, 2 * num_features_in, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(2 * num_features_in),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(2 * num_features_in, num_features_in, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(num_features_in),
            nn.ReLU()
        )

        self.down = nn.Sequential(
            nn.Conv2d(2 * num_features_in, feature_size, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        )

        if nn.Conv2d in self._modules:
            self._modules[nn.Conv2d].apply(self.init_weights)
        if nn.BatchNorm2d in self._modules:
            self._modules[nn.BatchNorm2d].apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x, training=True):
        # 残差模块降采样
        out = self.act(self.conv0(x) + x)
        out = self.down1(out)

        out = self.act(self.conv1(out) + out)
        out = self.conv2(out)
        out_ = out
        # 残差模块降采样
        out = self.down2(out)
        out = self.act(self.conv3(out) + out)
        out = self.conv4(out)
        out_ = out_ + self.up1(out)
        if training:
            x = x + self.up2(out_)
        out_ = self.down(out_)
        # 用1个k=1的卷积核把通道数降到1/8
        out = self.conv5(out)

        # out = out + out_
        return x, out_, out


def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

    return anchor_points


# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[0]) + 0.5) * stride
    shift_y = (np.arange(0, shape[1]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points


# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3, pc_pange=None, voxel_size=None):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        else:
            self.strides = strides

        self.row = row
        self.line = line
        self.pc_pange = pc_pange
        self.voxel_size = voxel_size

    def forward(self):
        image_shape = (int((self.pc_pange[3] - self.pc_pange[0]) / self.voxel_size[0]),
                       int((self.pc_pange[4] - self.pc_pange[1]) / self.voxel_size[1]))
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2 ** p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides, anchor_points) / 2
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False, num_anchor_points=4):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU(),
                ))
            fc_list.append(
                nn.Conv2d(input_channels, output_channels * num_anchor_points, kernel_size=3, stride=1, padding=1,
                          bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class Point2CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True, row=2, line=2):
        super().__init__()
        self.device = torch.device('cuda')
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = torch.as_tensor(np.round(grid_size).astype(np.int64), dtype=torch.int, device=self.device)
        self.point_cloud_range = torch.as_tensor(point_cloud_range, dtype=torch.float, device=self.device)
        self.voxel_size = torch.as_tensor(voxel_size, dtype=torch.float, device=self.device)
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.input_channels = self.model_cfg.get('INPUT_CHANNELS', 128)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        self.use_p2p_anchor = model_cfg.USE_P2P_ANCHOR
        self.heatmap_aux_stride = model_cfg.HEATMAP_AUX_STRIDE
        self.anchor_distance = model_cfg.ANCHOR_DISTANCE
        self.use_gt_mask = model_cfg.USE_GT_MASK
        self.time = []
        if self.use_p2p_anchor:
            num_anchor_points = row * line
            # 8800个anchor，从txt文件中读取anchor_points
            # point_xy = np.loadtxt('anchor_points_kitti.txt', dtype=np.float32)
            point_xy = np.loadtxt('anchor_points_waymo.txt', dtype=np.float32)
            # 画anchor的散点图
            # x = point_xy[:, 0]
            # y = point_xy[:, 1]
            # plt.figure(figsize=(20, 16), dpi=650)
            # plt.scatter(x, y, label='scatter figure', marker='.')
            # plt.legend()
            # plt.savefig('scatter.jpg')
            # plt.show()
            point_xy = torch.from_numpy(point_xy).to(self.device)
            zv = torch.tensor([5] * point_xy.shape[0], dtype=torch.float, device=self.device).unsqueeze(-1)
            self.anchor_points = torch.cat((point_xy, zv), dim=1).unsqueeze(0)

        else:
            # 如果使用普通anchor的版本
            num_anchor_points = 1
            yv, xv = torch.meshgrid(
                [torch.arange(0, grid_size[1] / self.feature_map_stride, self.anchor_distance,
                              dtype=torch.float, device=self.device),
                 torch.arange(0, grid_size[0] / self.feature_map_stride, self.anchor_distance, dtype=torch.float,
                              device=self.device)])
            zv = torch.tensor([grid_size[-1] / (self.feature_map_stride * 2)] * (yv.shape[0] * yv.shape[1]),
                              dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(-1)
            points = torch.stack((xv, yv), 2).view((1, (yv.shape[0] * yv.shape[1]), 2)).float()
            self.anchor_points = torch.cat((points, zv), dim=2)
            # # 增加一组8800个anchor的版本，给辅助分支使用
            # yv_, xv_ = torch.meshgrid(
            #     [torch.arange(0, grid_size[1] / self.feature_map_stride, self.anchor_distance // 2,
            #                   dtype=torch.float, device=self.device),
            #      torch.arange(0, grid_size[0] / self.feature_map_stride, self.anchor_distance // 2, dtype=torch.float,
            #                   device=self.device)])
            #
            # zv_ = torch.tensor([grid_size[-1] / (self.feature_map_stride * 2)] * (yv_.shape[0] * yv_.shape[1]),
            #                    dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(-1)
            # points_aux = torch.stack((xv_, yv_), 2).view((1, (yv_.shape[0] * yv_.shape[1]), 2)).float()
            # self.anchor_points_aux = torch.cat((points_aux, zv_), dim=2)

        self.shared_conv_res = SharedConvResV6(num_features_in=self.input_channels,
                                               feature_size=self.model_cfg.SHARED_CONV_CHANNEL)
        # self.shared_conv_res = SharedConvResV3(num_features_in=self.input_channels,
        #                                        feature_size=self.model_cfg.SHARED_CONV_CHANNEL)
        self.heatmap = Heatmap(num_classes=self.num_class)
        # self.densitymap = Densitymap()
        # self.mask = Mask()

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            self.heads_list.append(
                SeparateHead(
                    # input_channels=2 * self.model_cfg.SHARED_CONV_CHANNEL,
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    num_anchor_points=num_anchor_points
                )
            )

        self.heads_aux_list = nn.ModuleList()
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            self.heads_aux_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    num_anchor_points=num_anchor_points
                )
            )

        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

        # Loss parameters:
        cfg = self.model_cfg
        class_weight = cfg.CLASS_WEIGHT
        class_weight_cost = cfg.CLASS_WEIGHT_COST
        giou_weight = cfg.GIOU_WEIGHT
        l1_weight = cfg.L1_WEIGHT
        vel_weight = cfg.get('VEL_WEIGHT', 0.0)
        center_weight = cfg.CENTER_WEIGHT
        heatmap_weight = cfg.HEATMAP_WEIGHT
        density_weight = cfg.DENSITY_WEIGHT
        mask_weight = cfg.MASK_WEIGHT
        no_object_weight = cfg.NO_OBJECT_WEIGHT
        self.use_align = cfg.USE_ALIGN
        self.use_focal = cfg.USE_FOCAL
        self.use_fed_loss = cfg.USE_FED_LOSS
        self.use_ota = cfg.USE_OTA
        self.use_aux = cfg.USE_AUX
        self.test_aux_head = cfg.TEST_AUX_HEAD

        # Build Criterion.
        if self.use_ota:
            self.matcher = HungarianMatcherDynamicK(cfg=cfg,
                                                    cost_class=class_weight_cost,
                                                    cost_bbox=l1_weight,
                                                    cost_giou=giou_weight,
                                                    use_focal=self.use_focal)
        else:
            self.matcher = HungarianMatcher(cfg=cfg,
                                            cost_class=class_weight_cost,
                                            cost_bbox=l1_weight,
                                            cost_giou=giou_weight,
                                            use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight,
                       "loss_vel": vel_weight, "loss_center": center_weight, "loss_hm": heatmap_weight,
                       "loss_density": density_weight, "loss_mask": mask_weight, "loss_bbox_presudo": l1_weight}
        if self.use_aux:
            aux_weight_dict = {}
            for i in range(1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "hm"]

        if self.use_align:
            self.criterion = SetCriterion_align(cfg=cfg,
                                      num_classes=self.num_class,
                                      matcher=self.matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      grid_size=self.grid_size / self.feature_map_stride,
                                      pc_range=self.point_cloud_range)
        else:
            self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_class,
                                      matcher=self.matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal,
                                      grid_size=self.grid_size / self.feature_map_stride,
                                      pc_range=self.point_cloud_range)
        self.to(self.device)

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0,
                              max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

        return heatmap

    def assign_targets(self, gt_boxes, feature_map_size=None, feature_map_stride=8):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head,
                    feature_map_size=feature_map_size, feature_map_stride=feature_map_stride,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))

            heatmaps = torch.stack(heatmap_list, dim=0)
        return heatmaps

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        targets = self.forward_ret_dict['targets']
        output = self.forward_ret_dict['pred_dicts']
        loss_dict = self.criterion(output, targets)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]

        rpn_loss = sum(loss_dict.values())

        loss_dict['rpn_loss'] = rpn_loss.item()

        return rpn_loss, loss_dict

    def prepare_targets(self, gt_classes, gt_boxes, gt_vel):

        new_targets = []
        N = len(gt_classes)

        assert len(gt_classes) == len(gt_boxes)
        for b in range(N):

            cur_gt = gt_boxes[b]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[b][:cnt + 1].int()
            target = {}
            if gt_vel is not None:
                cur_gt_vel = gt_vel[b][:cnt + 1]
                target['gt_vel'] = cur_gt_vel
            gt_classes_per_image = torch.as_tensor(cur_gt_classes - 1, dtype=torch.long, device=self.device)
            gt_boxes_per_image = torch.as_tensor(cur_gt, dtype=torch.float, device=self.device)
            target["labels"] = gt_classes_per_image.to(self.device)
            grid_size_xyz = self.grid_size / self.feature_map_stride
            grid_size_tgt = grid_size_xyz.unsqueeze(0).repeat(len(gt_boxes_per_image), 1)
            target['grid_size_tgt'] = grid_size_tgt.to(self.device)
            target['grid_size_xyz'] = grid_size_xyz

            image_size_xyxy_tgt = (self.point_cloud_range[3:] - self.point_cloud_range[:3]).unsqueeze(0).repeat(
                len(gt_boxes_per_image), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)

            offset_tgt = torch.as_tensor(self.point_cloud_range[:3], dtype=torch.float, device=self.device)
            offset_tgt = offset_tgt.unsqueeze(0).repeat(len(gt_boxes_per_image), 1)
            target["offset_size_xyxy_tgt"] = offset_tgt.to(self.device)

            target['gt_boxes'] = gt_boxes_per_image
            new_targets.append(target)

        all_targets_dict = {
            'targets': new_targets,
        }
        return all_targets_dict

    def generate_predicted_boxes(self, batch_size, output_list):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()
        pred_boxes = output_list[0]['pred_boxes']
        pred_boxes[:, :, :3] = pred_boxes[:, :, :3] * self.voxel_size * self.feature_map_stride
        pred_boxes[:, :, 3:6] = pred_boxes[:, :, 3:6] * self.voxel_size * self.feature_map_stride
        pred_boxes[:, :, -1] = pred_boxes[:, :, -1]

        offset = self.point_cloud_range[:3].repeat(pred_boxes.shape[0], pred_boxes.shape[1], 1)
        pred_boxes[:, :, :3] = pred_boxes[:, :, :3] + offset
        if output_list[0]['vel'] is not None:
            pred_boxes = torch.cat([pred_boxes, output_list[0]['vel']], dim=-1)

        pred_scores = output_list[0]['pred_logits'].sigmoid()
        pred_logits = output_list[0]['pred_logits']

        # if output_list[0]['iou'] is not None:
        #     pred_iou = ((output_list[0]['iou'] + 1) * 0.5).flatten(0, 1).repeat(1, 3) ** torch.as_tensor(
        #         [0.68, 0.71, 0.65], device=pred_scores.device).clamp(0, 1)
        #     pred_scores = pred_scores ** torch.as_tensor([0.32, 0.29, 0.35], device=pred_scores.device)
        #     pred_scores = pred_scores * pred_iou.clamp(0, 1)

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
            'pred_logits': [],
        } for k in range(batch_size)]

        for b in range(batch_size):
            cur_pred_boxes = pred_boxes[b]
            cur_pred_scores = pred_scores[b]
            cur_pred_logits = pred_logits[b]
            score, cur_pred_labels = cur_pred_scores.topk(1, dim=-1, largest=True, sorted=False)
            cur_pred_scores = score.squeeze(1)
            cur_pred_labels = cur_pred_labels.squeeze(1)
            score_mask = cur_pred_scores > post_process_cfg.SCORE_THRESH
            cx_mask = (cur_pred_boxes[:, 0] > post_center_limit_range[0]) & (
                    cur_pred_boxes[:, 0] < post_center_limit_range[3])
            cy_mask = (cur_pred_boxes[:, 1] > post_center_limit_range[1]) & (
                    cur_pred_boxes[:, 1] < post_center_limit_range[4])
            cz_mask = (cur_pred_boxes[:, 2] > post_center_limit_range[2]) & (
                    cur_pred_boxes[:, 2] < post_center_limit_range[5])
            is_in_range = ((cx_mask.long() + cy_mask.long() + cz_mask.long()) == 3)
            box_mask = is_in_range & score_mask
            cur_pred_boxes = cur_pred_boxes[box_mask, :]
            cur_pred_scores = cur_pred_scores[box_mask]
            cur_pred_logits = cur_pred_logits[box_mask]
            cur_pred_labels = cur_pred_labels[box_mask]

            ret_dict[b]['pred_boxes'].append(cur_pred_boxes)
            ret_dict[b]['pred_scores'].append(cur_pred_scores)
            ret_dict[b]['pred_logits'].append(cur_pred_logits)
            ret_dict[b]['pred_labels'].append(cur_pred_labels)

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_logits'] = torch.cat(ret_dict[k]['pred_logits'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        # assert not torch.any(torch.isnan(spatial_features_2d))
        batch_size = spatial_features_2d.shape[0]
        # torch.cuda.synchronize()
        # start = time.time()
        if not self.training or self.predict_boxes_when_training or self.test_aux_head:
            spatial_features_2d, x_aux, x = self.shared_conv_res(spatial_features_2d, training=False)
        else:
            spatial_features_2d, x_aux, x = self.shared_conv_res(spatial_features_2d, training=True)
        # torch.cuda.synchronize()
        # print('shared_conv_res time: ', time.time() - start)
        # torch.cuda.synchronize()
        # start = time.time()

        anchor_points = self.anchor_points.repeat(batch_size, 1, 1)
        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))
        for i in pred_dicts[0].keys():
            if self.use_p2p_anchor:
                dim = int(pred_dicts[0][i].shape[1] / 4)  # 如果用了和p2pnet一样的anchor，这里要除以4
            else:
                dim = int(pred_dicts[0][i].shape[1])
            pred_dicts[0][i] = pred_dicts[0][i].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, dim)

        center = pred_dicts[0]['center'] + anchor_points
        output_class = pred_dicts[0]['class']
        # class_mask = (output_class[:, :, 0] == 0) & (output_class[:, :, 1] == 0) & (output_class[:, :, 2] == 0)
        # class_mask = class_mask.unsqueeze(dim=-1).repeat(1, 1, output_class.shape[-1])
        # output_class = output_class.masked_fill(class_mask, -1e2)
        size = pred_dicts[0]['dim'].clamp(max=10.0, min=1e-6).exp()

        rot_sin = pred_dicts[0]['rot'][:, :, 0].unsqueeze(dim=-1)
        rot_cos = pred_dicts[0]['rot'][:, :, 1].unsqueeze(dim=-1)
        angle = torch.atan2(rot_sin, rot_cos)

        vel = pred_dicts[0]['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
        # iou = F.sigmoid(pred_dicts[0]['iou']) if 'iou' in self.separate_head_cfg.HEAD_ORDER else None
        iou = pred_dicts[0]['iou'] if 'iou' in self.separate_head_cfg.HEAD_ORDER else None

        outputs_coord = torch.cat([center, size, angle], dim=-1)
        pred_dicts[0].update({'pred_logits': output_class})

        # # predict a mask to filter out the boxes that are not in the range of the gt_box
        # mask_maps_pred = self.mask(x)
        # if not self.training or self.predict_boxes_when_training:
        #     mask_pred = torch.softmax(mask_maps_pred, dim=1)
        #     mask_pred = torch.argmax(mask_pred, dim=1).view(batch_size, -1)
        #     mask_pred = mask_pred.sum(dim=1)
        #     mask_pred[(mask_pred > 0)] = 1
        #     outputs_coord[(mask_pred == 0), :] = 0
        #     pred_dicts[0]['pred_logits'][(mask_pred == 0), :] = -100
        # # draw the mask_pred
        # mask_pred = torch.argmax(mask_pred, dim=1)
        # mask_pred_ = mask_pred.cpu().numpy()[0, :, :]
        # mask_pred_ = mask_pred_ / np.max(mask_pred_) * 255
        # mask_pred_ = mask_pred_.astype(np.uint8)
        # mask_pred_ = cv2.applyColorMap(mask_pred_, 3)
        # cv2.imwrite('mask_pred.png', mask_pred_)
        # # draw the mask_gt
        # heatmaps_ = self.assign_targets(
        #     data_dict['gt_boxes'], feature_map_size=x.size()[2:], feature_map_stride=32)
        # mask_map_batch_list = []
        # for b in range(batch_size):
        #     map = heatmaps_[b].squeeze(0)
        #     map = ((map == 1).long().sum(dim=0) == 0).long().cpu().numpy().astype(np.uint8)
        #     map = cv2.distanceTransform(map, cv2.DIST_L2, 0)
        #     map[map > 3] = -1
        #     map[map >= 0] = 1
        #     map[map != 1] = 0
        #     mask_map_batch_list.append(torch.tensor(map, dtype=torch.float32).unsqueeze(0))
        # mask_gt = torch.as_tensor(torch.cat(mask_map_batch_list, dim=0), device=self.device)
        # mask_gt = mask_gt.cpu().numpy()[0, :, :]
        # mask_gt = mask_gt / np.max(mask_gt) * 255
        # mask_gt = mask_gt.astype(np.uint8)
        # mask_gt = cv2.applyColorMap(mask_gt, 3)
        # cv2.imwrite('mask_gt.png', mask_gt)

        # saved_index = torch.nonzero(mask_pred == 1, as_tuple=False)

        if self.training:
            outputs_coord_match = outputs_coord.detach().clone()
            if self.use_gt_mask:
                if data_dict['cur_epoch'] <= 10:
                    # 使用一个mask来辅助匹配
                    # # mask the box that is away from gt_box
                    heatmaps_ = self.assign_targets(
                        data_dict['gt_boxes'], feature_map_size=x.size()[2:],
                        feature_map_stride=self.feature_map_stride * 8)
                    mask_map_batch_list = []
                    mask_map_gt_batch_list = []
                    for b in range(batch_size):
                        map = heatmaps_[b].squeeze(0)
                        # Waymo单类mask
                        # map = heatmaps_[b]
                        # map[0, (map[0, :, :] == 0)] = 3
                        # map[0, (map[0, :, :] != 3)] = 0
                        # map[1, (map[1, :, :] == 0)] = 3
                        # map[1, (map[1, :, :] != 3)] = 1
                        # map[2, (map[2, :, :] == 0)] = 3
                        # map[2, (map[2, :, :] != 3)] = 2
                        map = ((map == 1).long().sum(dim=0) == 0).long().cpu().numpy().astype(np.uint8)
                        map = cv2.distanceTransform(map, cv2.DIST_L2, 0)
                        i = 1.45
                        # i = 4.25  # 3*根号2
                        # i = 7.08  # 5*根号2
                        map[map > i] = -1
                        map[map >= 0] = 1
                        map[map != 1] = 0
                        # mask_map_batch_list.append(torch.tensor(map, dtype=torch.float32).unsqueeze(0))
                        # map[(map != 3)] = 1
                        # map[(map == 3)] = 0
                        # map = map.sum(dim=0)
                        # map[map > 0] = 1
                        mask_map_gt_batch_list.append(torch.tensor(map, dtype=torch.float32).unsqueeze(0))
                    # mask_maps = torch.as_tensor(torch.cat(mask_map_batch_list, dim=0), device=self.device)
                    mask_maps_gt = torch.as_tensor(torch.cat(mask_map_gt_batch_list, dim=0), device=self.device).view(
                        batch_size, -1)
                    # mask_pred = mask_maps_gt.view(batch_size, -1)
                    outputs_coord_match[(mask_maps_gt == 0), :] = 0

            output = {'pred_logits': pred_dicts[0]['pred_logits'], 'pred_boxes': outputs_coord,
                      'pred_boxes_match': outputs_coord_match, 'anchor_points': anchor_points,
                      'pred_center_offsets': pred_dicts[0]['center'],
                      'pred_size': size, 'pred_rot': torch.cat([rot_sin, rot_cos], dim=-1),
                      'vel': vel, 'iou': iou, 'cur_epoch': data_dict['cur_epoch'], 'use_mto': False}

            heatmap_pred = self.heatmap(spatial_features_2d)
            heatmaps = self.assign_targets(data_dict['gt_boxes'],
                                           # feature_map_size=[i * 2 for i in spatial_features_2d.size()[2:]],
                                           feature_map_size=spatial_features_2d.size()[2:],
                                           feature_map_stride=self.heatmap_aux_stride)

            if self.use_aux:
                # anchor_points_aux = self.anchor_points_aux.repeat(batch_size, 1, 1)
                pred_dicts_aux = []

                for head in self.heads_aux_list:
                    # for head in self.heads_list:
                    pred_dicts_aux.append(head(x_aux))
                for i in pred_dicts_aux[0].keys():
                    if self.use_p2p_anchor:
                        dim = int(pred_dicts_aux[0][i].shape[1] / 4)  # 如果用了和p2pnet一样的anchor，这里要除以4
                    else:
                        dim = int(pred_dicts_aux[0][i].shape[1])
                    pred_dicts_aux[0][i] = pred_dicts_aux[0][i].permute(0, 2, 3, 1).contiguous().view(batch_size, -1,
                                                                                                      dim)

                center_aux = pred_dicts_aux[0]['center'] + anchor_points
                # center_aux = center_aux + anchor_points_aux
                output_class_aux = pred_dicts_aux[0]['class']
                # class_mask = (output_class_aux[:, :, 0] == 0) & (output_class_aux[:, :, 1] == 0) & (
                #         output_class_aux[:, :, 2] == 0)
                # class_mask = class_mask.unsqueeze(dim=-1).repeat(1, 1, output_class_aux.shape[-1])
                # output_class_aux = output_class_aux.masked_fill(class_mask, -1e2)
                size_aux = pred_dicts_aux[0]['dim'].clamp(max=10.0, min=1e-6).exp()

                rot_sin_aux = pred_dicts_aux[0]['rot'][:, :, 0].unsqueeze(dim=-1)
                rot_cos_aux = pred_dicts_aux[0]['rot'][:, :, 1].unsqueeze(dim=-1)
                angle_aux = torch.atan2(rot_sin_aux, rot_cos_aux)

                vel_aux = pred_dicts_aux[0]['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
                iou_aux = F.sigmoid(pred_dicts_aux[0]['iou']) if 'iou' in self.separate_head_cfg.HEAD_ORDER else None

                outputs_coord_aux = torch.cat([center_aux, size_aux, angle_aux], dim=-1)
                outputs_coord_aux_match = outputs_coord_aux.detach().clone()
                pred_dicts_aux[0].update({'pred_logits': output_class_aux})

                # # mask the box that is away from gt_box in aux_output
                # outputs_coord_aux[(mask_pred == 0), :] = 0
                # pred_dicts_aux[0]['pred_logits'][(mask_pred == 0), :] = -100
                # 如果在不同epoch中用不同的mask
                # if data_dict['cur_epoch'] < 70:
                #     mask_pred = mask_maps_gt.view(batch_size, -1)
                #     outputs_coord_aux[(mask_pred == 0), :] = 0
                #     pred_dicts_aux[0]['pred_logits'][(mask_pred == 0), :] = -100
                # else:
                #     mask_pred = torch.softmax(mask_maps_pred, dim=1)
                #     mask_pred = torch.argmax(mask_pred, dim=1).view(batch_size, -1)
                #     mask_pred = mask_pred.sum(dim=1)
                #     mask_pred[(mask_pred > 0)] = 1
                #     outputs_coord[(mask_pred == 0), :] = 0
                #     pred_dicts[0]['pred_logits'][(mask_pred == 0), :] = -100
                # # 前10个epooch使用gt mask
                if self.use_gt_mask:
                    if data_dict['cur_epoch'] <= 10:
                        mask_pred = mask_maps_gt.view(batch_size, -1)
                        outputs_coord_aux_match[(mask_pred == 0), :] = 0

                output_aux = [{'pred_logits': pred_dicts_aux[0]['pred_logits'], 'pred_boxes': outputs_coord_aux,
                               'pred_boxes_match': outputs_coord_aux_match, 'anchor_points': anchor_points,
                               'pred_center_offsets': pred_dicts_aux[0]['center'],
                               'pred_size': size_aux, 'pred_rot': torch.cat([rot_sin_aux, rot_cos_aux], dim=-1),
                               'vel': vel_aux, 'iou': iou_aux, 'cur_epoch': data_dict['cur_epoch'],
                               'use_mto': False}]
                output.update({'aux_outputs': output_aux})

            # # draw heatmap_gt
            # heatmaps_ = self.assign_targets(
            #     data_dict['gt_boxes'], feature_map_size=x.size()[2:], feature_map_stride=32)
            # density_map_batch_list = []
            # for b in range(batch_size):
            #     map = heatmaps_[b].squeeze(0)
            #     map = ((map == 1).long().sum(dim=0) == 0).long().cpu().numpy().astype(np.uint8)
            #     map = cv2.distanceTransform(map, cv2.DIST_L2, 0)
            #     map[map > 3] = -1
            #     map[map >= 0] = 1
            #     map[map != 1] = 0
            #     density_map_batch_list.append(torch.tensor(map, dtype=torch.float32).unsqueeze(0))
            # density_maps = torch.as_tensor(torch.cat(density_map_batch_list, dim=0), device=self.device).view(
            #     batch_size, -1)
            # density_map = density_maps.cpu().numpy()[0, 0, :, :]
            # density_map = density_map / np.max(density_map) * 255
            # density_map = density_map.astype(np.uint8)
            # density_map = cv2.applyColorMap(density_map, 3)
            # cv2.imwrite('density_map.png', density_map)

            if 'vel' in self.separate_head_cfg.HEAD_ORDER:
                targets_dict = self.prepare_targets(
                    gt_classes=data_dict['gt_boxes'][:, :, -1],
                    gt_boxes=data_dict['gt_boxes'][:, :, :7],
                    gt_vel=data_dict['gt_boxes'][:, :, 7:9]
                )
            else:
                targets_dict = self.prepare_targets(
                    gt_classes=data_dict['gt_boxes'][:, :, -1],
                    gt_boxes=data_dict['gt_boxes'][:, :, :7],
                    gt_vel=None
                )
            targets_dict['targets'][0].update({'heatmaps': heatmaps})
            # targets_dict['targets'][0].update({'mask_maps': mask_maps})
            output.update({'heatmaps_pred': heatmap_pred})
            output.update({'grid_anchor': anchor_points})

            self.forward_ret_dict.update(targets_dict)
            # targets_dict['targets'][0].update({'density_maps': density_maps})
            # output.update({'density_map_pred': density_map_pred})

            self.forward_ret_dict['pred_dicts'] = output

        if not self.training or self.predict_boxes_when_training:

            # torch.cuda.synchronize()
            # print('head time: ', time.time() - start)
            # torch.cuda.synchronize()
            # start = time.time()

            if not self.test_aux_head:
                output_list = [
                    {'pred_logits': pred_dicts[0]['pred_logits'], 'pred_boxes': outputs_coord, 'vel': vel, 'iou': iou}]
                pred_dicts = self.generate_predicted_boxes(
                    data_dict['batch_size'], output_list)
                #
                # torch.cuda.synchronize()
                # print('postprocess time: ', time.time() - start)
            else:
                pred_dicts_aux = []
                for head in self.heads_aux_list:
                    # for head in self.heads_list:
                    pred_dicts_aux.append(head(x_aux))
                for i in pred_dicts_aux[0].keys():
                    if self.use_p2p_anchor:
                        dim = int(pred_dicts_aux[0][i].shape[1] / 4)  # 如果用了和p2pnet一样的anchor，这里要除以4
                    else:
                        dim = int(pred_dicts_aux[0][i].shape[1])
                    pred_dicts_aux[0][i] = pred_dicts_aux[0][i].permute(0, 2, 3, 1).contiguous().view(batch_size, -1,
                                                                                                      dim)
                center_aux = pred_dicts_aux[0]['center'] + anchor_points
                # center_aux = center_aux + anchor_points_aux
                output_class_aux = pred_dicts_aux[0]['class']
                size_aux = pred_dicts_aux[0]['dim'].clamp(max=10.0).exp()

                rot_sin_aux = pred_dicts_aux[0]['rot'][:, :, 0].unsqueeze(dim=-1)
                rot_cos_aux = pred_dicts_aux[0]['rot'][:, :, 1].unsqueeze(dim=-1)
                angle_aux = torch.atan2(rot_sin_aux, rot_cos_aux)
                vel_aux = pred_dicts_aux[0]['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
                outputs_coord_aux = torch.cat([center_aux, size_aux, angle_aux], dim=-1)
                pred_dicts_aux[0].update({'pred_logits': output_class_aux})
                output_list_aux = [{'pred_logits': pred_dicts_aux[0]['pred_logits'], 'pred_boxes': outputs_coord_aux}]
                pred_dicts = self.generate_predicted_boxes(
                    data_dict['batch_size'], output_list_aux
                )

            # # 计算预测的每个类别的box数量
            # pred_boxes_num = []
            # for i in range(len(pred_dicts)):
            #     pred_num_ = {}
            #     # compute the number of each class
            #     for key in pred_dicts[i]['pred_labels'].tolist():
            #         pred_num_[key] = pred_num_.get(key, 0) + 1
            #     pred_num = []
            #     for index, _ in enumerate(self.class_names):
            #         pred_num.append(pred_num_.get(index + 1, 0))
            #     pred_boxes_num.append(pred_num)
            #
            # data_dict['pred_num'] = pred_boxes_num

            # heatmaps = self.assign_targets(
            #     data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
            #     feature_map_stride=8)
            # density_map_list = []
            # density_map_batch_list = []
            # for b in range(batch_size):
            #     map = heatmaps[b].squeeze(0)
            #     map = (map == 1).long()
            #     for i in range(map.shape[0]):
            #         density_map = map[i].squeeze(0).cpu().numpy().astype(np.float32)
            #         density_map = gaussian_filter(density_map, 4)
            #         density_map = torch.from_numpy(density_map).unsqueeze(0).to(self.device)
            #         density_map_list.append(density_map)
            #     density_map_batch = torch.cat(density_map_list, dim=0).unsqueeze(0)
            #     density_map_list = []
            #     density_map_batch_list.append(density_map_batch)
            # density_maps = torch.cat(density_map_batch_list, dim=0)
            # density_maps = density_maps.cpu().numpy()[0, 0, :, :]
            # density_maps = density_maps / np.max(density_maps) * 255
            # density_maps = density_maps.astype(np.uint8)
            # density_maps = cv2.applyColorMap(density_maps, 3)
            #
            # cv2.imwrite('density_maps.png', density_maps)
            #
            # # save predicted density map sum in data_dict
            # density_map_pred = self.densitymap(spatial_features_2d)
            # density_map_sum_pred_batch = []
            # density_map_sum_pred = []
            # for b in range(batch_size):
            #     for index, key in enumerate(self.class_names):
            #         density_map_sum_pred_batch.append(float(density_map_pred[b][index].sum()))
            #     density_map_sum_pred_batch = torch.tensor(density_map_sum_pred_batch).unsqueeze(0)
            #     density_map_sum_pred.append(density_map_sum_pred_batch)
            #     density_map_sum_pred_batch = []
            # density_map_sum_pred = torch.cat(density_map_sum_pred, dim=0)
            # data_dict['density_map_sum_pred'] = density_map_sum_pred
            #
            # # save predicted density map
            # density_map_pred = density_map_pred.cpu().numpy()[0, 0, :, :]
            # density_map_pred = density_map_pred / np.max(density_map_pred) * 255
            # density_map_pred = density_map_pred.astype(np.uint8)
            # density_map_pred = cv2.applyColorMap(density_map_pred, 3)
            #
            # cv2.imwrite('density_map_pred.png', density_map_pred)

            # output_pred_boxes = []
            # output_pred_logits = []
            # output_ = []
            # save = pred_dicts[0]['pred_boxes'].clone()
            # pred_dicts[0]['pred_boxes'] = pred_dicts[0]['pred_boxes'].unsqueeze(0)
            # pred_dicts[0]['pred_boxes'][:, :, 0] = pred_dicts[0]['pred_boxes'][:, :, 0] / 0.2
            # pred_dicts[0]['pred_boxes'][:, :, 1] = (pred_dicts[0]['pred_boxes'][:, :, 1] + 40) / 0.2
            # pred_dicts[0]['pred_boxes'][:, :, 2] = (pred_dicts[0]['pred_boxes'][:, :, 2] + 3) / 0.4
            # pred_dicts[0]['pred_boxes'][:, :, 3] = pred_dicts[0]['pred_boxes'][:, :, 3] / 0.2
            # pred_dicts[0]['pred_boxes'][:, :, 4] = pred_dicts[0]['pred_boxes'][:, :, 4] / 0.2
            # pred_dicts[0]['pred_boxes'][:, :, 5] = pred_dicts[0]['pred_boxes'][:, :, 5] / 0.4
            #
            # pred_dicts[0]['pred_logits'] = pred_dicts[0]['pred_logits'].unsqueeze(0)
            # pred_dicts[0]['pred_labels'] = pred_dicts[0]['pred_labels'].unsqueeze(0)
            # pred_dicts[0]['pred_scores'] = pred_dicts[0]['pred_scores'].unsqueeze(0)
            # targets_dict = self.prepare_targets(
            #     gt_classes=data_dict['gt_boxes'][:, :, -1],
            #     gt_boxes=data_dict['gt_boxes'][:, :, :-1],
            # )
            # targets = targets_dict['targets']
            # if len(targets[0]['labels']) == 0 or pred_dicts[0]["pred_logits"].shape[1] == 0:
            #     pred_dicts[0]['pred_boxes'] = pred_dicts[0]['pred_boxes'].squeeze(0)
            #     pred_dicts[0]['pred_logits'] = pred_dicts[0]['pred_logits'].squeeze(0)
            #     pred_dicts[0]['pred_labels'] = pred_dicts[0]['pred_labels'].squeeze(0)
            #     pred_dicts[0]['pred_scores'] = pred_dicts[0]['pred_scores'].squeeze(0)
            #     # pass
            # else:
            #     indices = self.matcher(pred_dicts[0], targets)
            #     pred_dicts[0]['pred_logits'] = pred_dicts[0]['pred_logits'].squeeze(0)
            #     pred_dicts[0]['pred_labels'] = pred_dicts[0]['pred_labels'].squeeze(0)
            #     pred_dicts[0]['pred_scores'] = pred_dicts[0]['pred_scores'].squeeze(0)
            #     pred_dicts[0]['pred_boxes'] = save
            #     data_dict.update({'indices': indices})
            #     batch_size = data_dict['batch_size']
            #     for i in range(batch_size):
            #         icdice_pair = indices[i]
            #         out_indice = icdice_pair[0]
            #         gt_indice = icdice_pair[1]
            #         if targets[i]['labels'].equal(-torch.ones(1).to(self.device)):
            #             output = output
            #         else:
            #             for j in range(len(out_indice)):
            #                 # output_pred_boxes.append(output['pred_boxes'][i, out_indice[j], :].unsqueeze(0).unsqueeze(0))
            #                 # output_pred_logits.append(output['pred_logits'][i, out_indice[j], :].unsqueeze(0).unsqueeze(0))
            #                 center = targets[i]['gt_boxes'][gt_indice[j], :3].clone()
            #                 size = targets[i]['gt_boxes'][gt_indice[j], 3:6].clone()
            #                 angle = targets[i]['gt_boxes'][gt_indice[j], -1].clone()
            #                 lable = targets[i]['labels'][gt_indice[j]].clone()
            #                 # pred_dicts[0]['pred_boxes'][out_indice[j], :][0] = center[0]
            #                 # pred_dicts[0]['pred_boxes'][out_indice[j], :][1] = center[1]
            #                 # pred_dicts[0]['pred_boxes'][out_indice[j], :][2] = center[2]
            #                 # pred_dicts[0]['pred_boxes'][out_indice[j], :][:3] = center
            #                 # pred_dicts[0]['pred_boxes'][out_indice[j], :][3:6] = size
            #                 pred_dicts[0]['pred_boxes'][out_indice[j], :][-1] = angle
            #                 # pred_dicts[0]['pred_labels'][out_indice[j]] = (lable + 1)
            #
            # # pred_boxes = torch.stack(output_pred_boxes, dim=1).to(self.device)
            # # pred_logits = torch.stack(output_pred_logits, dim=1).to(self.device)
            # # output['pred_boxes'] = pred_boxes
            # # output['pred_logits'] = pred_logits

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict
