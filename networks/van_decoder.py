from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *

from .da_att import *



class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv

        sasc_output = self.conv8(feat_sum)
        
        return sasc_output

## SpatialAttetion
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, in_feature):
        x = in_feature
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #x = avg_out
        #x = max_out
        x = self.conv1(x)
        return self.sigmoid(x).expand_as(in_feature) * in_feature
        #return self.sigmoid(x).expand_as(in_feature) * in_feature + in_feature

## ChannelAttetion
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        #self.conv1 = nn.Conv2d(in_planes, out_planes, 1)
        #self.act1 = nn.ReLU(inplace = True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_planes,in_planes // ratio, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_planes // ratio, in_planes, bias = False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(in_planes, out_planes, 3, padding=1)
        self.act3 = nn.ReLU(inplace = True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        #x1 = self.act1(self.conv1(in_feature))
        x1 = in_feature
        b, c, _, _ = x1.size()
        avg_out = self.fc(self.avg_pool(x1).view(b,c)).view(b, c, 1, 1)
        
        return self.act3(self.conv3(self.sigmoid(avg_out).expand_as(x1) * in_feature))

    

class VANDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), height=192, width=640, num_output_channels=1, use_skips=True):
        super(VANDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 320])

        # decoder
        self.convs = OrderedDict()
        self.convs[("da")] = DANetHead(self.num_ch_enc[-1], self.num_ch_dec[-1])
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_dec[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            if self.use_skips and i > 1:
                self.convs[("sa", i)] = SpatialAttention(kernel_size=7)
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 1:
                num_ch_in += self.num_ch_enc[i - 2]
            num_ch_out = self.num_ch_dec[i]
            #self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs[("upconv", i, 1)] = ChannelAttention(num_ch_in, num_ch_out, ratio=4)

        for s in range(3, -1, -1):
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = self.convs[("da")](input_features[-1])

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            
            x = [upsample(x)]
            if self.use_skips and i > 1:
                x += [self.convs[("sa", i)](input_features[i - 2])]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
