# coding=utf-8
import pdb

import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder

from PIL import Image
import cv2
import random
import os
# import neuron
# import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from model_net import *
from conv_dnm import *
# from spikingjelly.clock_driven import neuron, surrogate, functional
from spikingjelly.activation_based import neuron, layer
from spikingjelly.clock_driven import surrogate, functional

# from spikingjelly.activation_based import neuron, surrogate,  functional
# from torchinfo import summary
bn_momentum = 0.1


class ConvBlock0(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, dilation_rate=(2, 2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock1(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.ReLU(nplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock2(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dilation_rate=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=dilation_rate,
                      dilation=(dilation_rate, dilation_rate),
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.ReLU(nplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class soout(nn.Module):
    def __init__(self, in_channels, size):
        super(soout, self).__init__()
        self.conv = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=size),
            nn.Conv2d(in_channels, 1, kernel_size=1, padding=1, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ICModel(nn.Module):
    def __init__(self, input_size, out_size):
        super(ICModel, self).__init__()

        self.out_size = out_size
        self.linear = nn.Conv2d(input_size, out_size, kernel_size=1)

    def forward(self, x):
        x1 = torch.sum(x, 1).unsqueeze(1)  # [batch, 1, H, W]
        x = self.linear(x)  # [batch, out_size, H, W]
        x = torch.relu(x)
        x = x + x1.repeat(1, self.out_size, 1, 1)
        return x

    def reset_parameters(self):
        pass  # No need to initialize parameters for 1x1 convolution layer


class ChannelReduction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelReduction, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
            # neuron.IFNode(surrogate_function=surrogate.ATan())
            # IFNode(surrogate_function=surrogate.ATan())
        )

    def forward(self, x):
        spikes = self.conv(x)
        spikes = neuron.surrogate.functional.to_spike(torch.relu(spikes))
        feats = neuron.surrogate.functional.spike_count(spikes)

        feats = feats.mean(dim=1, keepdim=True)

        return feats


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.if1 = ChannelReduction(64, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 8 * 8, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.if1(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class IFNode(nn.Module):
    def __init__(self, in_channels):
        super(IFNode, self).__init__()
        self.membrane = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        spikes = neuron.functional.izhikevich(self.membrane, x)
        self.membrane += spikes
        return spikes


class RRCNet(nn.Module):
    def __init__(self, device=DEVICE, m=5, flagMRNET=1, flagFRNet=1):
        super(RRCNet, self).__init__()

        # encoder
        self.enco1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

        print("Build enceder done..")

        # between encoder and decoder
        self.midco = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

        self.out7 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=32),
            nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )
        # decoder

        self.deco1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.out6 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=16),
            nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )

        self.deco2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.out5 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=8),
            nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )

        self.deco3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.out4 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(128, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )

        self.deco4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.out3 = nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.ReLU(nplace=True),
        )

        self.deco5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            # nn.BatchNorm2d(1, momentum=bn_momentum),
            # nn.ReLU(),
        )
        self.mrnet = MRNet(device=DEVICE, M=m, flag=flagMRNET)
        self.frnet = FRNet(device=DEVICE, M=m, flag=flagFRNet)

    def forward(self, x):
        id = []
        # encoder
        x = self.enco1(x)  # 1-2
        conv_2 = x
        x, id1 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)  # 保留最大值的位置
        id.append(id1)
        x = self.enco2(x)  # 2-4
        conv_4 = x
        x, id2 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id2)
        x = self.enco3(x)  # 5-7
        conv_7 = x

        x, id3 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id3)
        x = self.enco4(x)  # 8-10
        conv_10 = x
        x, id4 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id4)
        x = self.enco5(x)  # 11-13
        conv_13 = x
        x, id5 = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1, return_indices=True)
        id.append(id5)

        # between encoder and decoder
        x = self.midco(x)  # 14-16
        Out7 = self.out7(x)

        # decoder
        x = F.max_unpool2d(x, id[4], kernel_size=2)
        concat_1 = torch.cat((x, conv_13), axis=1)  # 拼接16池化后的结果和13

        x = self.deco1(concat_1)  # 17-19
        Out6 = self.out6(x)
        x = F.max_unpool2d(x, id[3], kernel_size=(2, 2))
        concat_2 = torch.cat((x, conv_10), axis=1)  # 拼接19池化后的结果和10

        x = self.deco2(concat_2)  # 20-22
        Out5 = self.out5(x)
        x = F.max_unpool2d(x, id[2], kernel_size=(2, 2))
        concat_3 = torch.cat((x, conv_7), axis=1)  # 拼接22池化后的结果和7

        x = self.deco3(concat_3)  # 23-25
        Out4 = self.out4(x)
        x = F.max_unpool2d(x, id[1], kernel_size=(2, 2))
        concat_4 = torch.cat((x, conv_4), axis=1)  # 拼接25池化后的结果和4

        x = self.deco4(concat_4)  # 26-27
        Out3 = self.out3(x)
        x = F.max_unpool2d(x, id[0], kernel_size=(2, 2))
        concat_5 = torch.cat((x, conv_2), axis=1)  # 拼接27池化后的结果和2

        x = self.deco5(concat_5)  # 28-29
        x = F.sigmoid(x)
        Out2 = x

        Out0, out_m = self.mrnet(Out2)
        Out1, out_f = self.frnet(Out0)

        return Out0, Out1, Out2, Out3, Out4, Out5, Out6, Out7
        # out2 + out_m - out_f == out1 (2+9-8=1)  ,out_f,out_m


class MRNet(nn.Module):
    def __init__(self, device=DEVICE, M=5, flag=1):
        super(MRNet, self).__init__()
        self.conv1 = ConvBlock1(1, 64)
        self.conv2 = ConvBlock2(64, 64, None, 2)  # in out mid dilation
        self.conv3 = ConvBlock2(64, 64, None, 3)
        self.conv4 = ConvBlock2(64, 64, None, 4)
        self.conv5 = ConvBlock2(64, 64, None, 3)
        self.conv6 = ConvBlock2(128, 64, None, 3)
        self.conv7 = ConvBlock2(64, 64, None, 2)
        self.conv8 = ConvBlock2(128, 64, None, 2)
        self.conv9 = ConvBlock1(64, 64)
        self.conv10 = ConvBlock1(128, 64)
        self.conv2d = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
        self.Dnm_conv2d = conv_dnm(64, 1, M)
        # ICModel
        self.ICModel_conv = ICModel(64, 1)
        # SNU
        self.conv_snu = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.if1 = ChannelReduction(64, 1)

        self.flag = flag

    def forward(self, x):
        input_size = x.size(0)  # batch_size
        data_x = x
        x = self.conv1(x)
        temp1 = x
        x = self.conv2(x)
        temp2 = x
        x = self.conv3(x)
        temp3 = x
        x = self.conv4(x)

        x = self.conv5(x)
        x = torch.cat((x, temp3), axis=1)  # 拼接3和7
        x = self.conv6(x)

        x = self.conv7(x)
        x = torch.cat((x, temp2), axis=1)  # 拼接2和8
        x = self.conv8(x)

        x = self.conv9(x)
        x = torch.cat((x, temp1), axis=1)  # 拼接1和9
        x = self.conv10(x)
        # pdb.set_trace()
        if self.flag == 1:
            x = self.Dnm_conv2d(x)
        elif self.flag == 2:
            x = self.ICModel_conv(x)
        elif self.flag == 3:
            x = self.conv_snu(x)
            x = self.bn1(x)
            x = self.if1(x)
        else:
            x = self.conv2d(x)

        out = torch.add(data_x, x)
        out0 = F.sigmoid(out)
        return out0, F.sigmoid(x)


class FRNet(nn.Module):
    def __init__(self, device=DEVICE, M=5, flag=1):
        super(FRNet, self).__init__()
        self.conv1 = ConvBlock1(1, 64)
        self.conv2 = ConvBlock1(64, 64)
        self.conv3 = ConvBlock1(64, 128)
        self.conv4 = ConvBlock1(128, 128)
        self.conv5 = ConvBlock1(128, 256)
        self.conv6 = ConvBlock1(384, 128)
        self.conv7 = ConvBlock1(256, 128)
        self.conv8 = ConvBlock1(192, 64)
        self.conv9 = ConvBlock1(128, 64)
        self.Dnm_conv2d = conv_dnm(64, 1, M)
        self.ICModel_conv = ICModel(64, 1)
        self.conv2d = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
        self.maxpool = nn.MaxPool2d(2)
        self.upsampling = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.flag = flag

    def forward(self, x):
        # input_size = x.size(0)  # batch_size
        data_x = x
        x = self.conv1(x)
        Conv1 = x
        x = self.maxpool(x)

        x = self.conv2(x)
        Conv2 = x
        x = self.maxpool(x)

        x = self.conv3(x)
        Conv3 = x
        x = self.maxpool(x)

        x = self.conv4(x)
        Conv4 = x
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.upsampling(x)
        x = torch.cat((x, Conv4), axis=1)

        x = self.conv6(x)
        x = self.upsampling(x)
        x = torch.cat((x, Conv3), axis=1)  # 拼接3和7

        x = self.conv7(x)
        x = self.upsampling(x)
        x = torch.cat((x, Conv2), axis=1)  # 拼接2和8

        x = self.conv8(x)
        x = self.upsampling(x)
        x = torch.cat((x, Conv1), axis=1)  # 拼接1和9

        x = self.conv9(x)
        if self.flag == 1:
            x = self.Dnm_conv2d(x)
        elif self.flag == 2:
            x = self.ICModel_conv(x)
        elif self.flag == 3:
            x = self.conv_snu(x)
            x = self.bn1(x)
            x = self.if1(x)
        else:
            x = self.conv2d(x)
        out = torch.subtract(data_x, x)
        out0 = F.sigmoid(out)

        return out0, x

# model = RRCNet()
# # print(model)
# #
# summary(model, (3, 384, 384))
