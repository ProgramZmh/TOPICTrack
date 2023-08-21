#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import time
import math
import numpy as np

from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
from .base_model import BaseModel

try:
    from .DCNv2.dcn_v2 import DCN
except:
    print('Import DCN failed')
    DCN = None

BN_MOMENTUM = 0.1


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1,
                        padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f, node_type=(DeformConv, DeformConv)):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = node_type[0](c, o)
            node = node_type[1](o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None,
                 node_type=DeformConv):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j],
                          node_type=node_type))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class GlobalConv(nn.Module):
    def __init__(self, chi, cho, k=7, d=1):
        super(GlobalConv, self).__init__()
        gcl = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=(k, 1), stride=1, bias=False,
                      dilation=d, padding=(d * (k // 2), 0)),
            nn.Conv2d(cho, cho, kernel_size=(1, k), stride=1, bias=False,
                      dilation=d, padding=(0, d * (k // 2))))
        gcr = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=(1, k), stride=1, bias=False,
                      dilation=d, padding=(0, d * (k // 2))),
            nn.Conv2d(cho, cho, kernel_size=(k, 1), stride=1, bias=False,
                      dilation=d, padding=(d * (k // 2), 0)))
        fill_fc_weights(gcl)
        fill_fc_weights(gcr)
        self.gcl = gcl
        self.gcr = gcr
        self.act = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.gcl(x) + self.gcr(x)
        x = self.act(x)
        return x


class Conv(nn.Module):
    def __init__(self, chi, cho):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# TODO：加一个dict，通过名字【eg：yolo-tiny】来制定channels和depth
# yolo_spec = {"yolo-tiny": (0.33,0.33),}


DLA_NODE = {
    'dcn': (DeformConv, DeformConv),
    'gcn': (Conv, GlobalConv),
    'conv': (Conv, Conv),
}


class CSPDarknetDLA(BaseModel):
    def __init__(self, num_layers, heads, head_convs, opt):
        assert head_convs['hm'][0] in [64, 256]
        super(CSPDarknetDLA, self).__init__(
            heads, head_convs, 1, head_convs['hm'][0], opt=opt)

        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]

        # yolo params: s
        wid_mul = 0.5
        dep_mul = 0.33
        self.out_features = ("dark3", "dark4", "dark5")
        depthwise = False
        act = "silu"

        # DCN params
        self.deconv_with_bias = False
        self.inplanes = 64

        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        out_channels = [base_channels,
                        base_channels*2,  # !must be 64
                        base_channels*4,
                        base_channels*8,
                        base_channels*16,
                        ]
        # stem
        self.stem = Focus(3, out_channels[0], ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, out_channels[1], 3, 2, act=act),
            CSPLayer(
                out_channels[1],
                out_channels[1],
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, out_channels[2], 3, 2, act=act),
            CSPLayer(
                out_channels[2],
                out_channels[2],
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, out_channels[3], 3, 2, act=act),
            CSPLayer(
                out_channels[3],
                out_channels[3],
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, out_channels[4], 3, 2, act=act),
            SPPBottleneck(out_channels[4],
                          out_channels[4], activation=act),
            CSPLayer(
                out_channels[4],
                out_channels[4],
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

        down_ratio = 4
        self.first_level = int(np.log2(down_ratio)) - 1
        self.last_level = 4
        channels = out_channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level],
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)

    def img2feats(self, x):
        output = []

        # backbone
        x = self.stem(x)
        output.append(x)
        x = self.dark2(x)
        output.append(x)
        x = self.dark3(x)
        output.append(x)
        x = self.dark4(x)
        output.append(x)
        x = self.dark5(x)
        output.append(x)

        x = self.dla_up(output)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]

    def img2feats_prev(self, x):
        output = []

        # backbone
        x = self.stem(x)
        output.append(x)
        x = self.dark2(x)
        output.append(x)
        x = self.dark3(x)
        output.append(x)
        x = self.dark4(x)
        output.append(x)
        x = self.dark5(x)
        output.append(x)

        x = self.dla_up(output)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1].detach()]


class CSPDarknetDCN(BaseModel):
    def __init__(self, num_layers, heads, head_convs, opt):
        assert head_convs['hm'][0] in [64, 256]
        super(CSPDarknetDCN, self).__init__(
            heads, head_convs, 1, head_convs['hm'][0], opt=opt)

        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]

        # TODO：封装
        # yolo params: s
        # wid_mul = 0.5
        # dep_mul = 0.33
        # yolo params: tiny
        wid_mul = 0.375
        dep_mul = 0.33
        self.out_features = ("dark3", "dark4", "dark5")
        depthwise = False
        act = "silu"

        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        out_channels = [base_channels,
                        base_channels*2,
                        base_channels*4,
                        base_channels*8,
                        base_channels*16,
                        ]

        # DCN params
        self.deconv_with_bias = False
        # backbone的最终channels要与DCN的第一层输入channels相等
        self.inplanes = out_channels[-1]

        # stem
        self.stem = Focus(3, out_channels[0], ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, out_channels[1], 3, 2, act=act),
            CSPLayer(
                out_channels[1],
                out_channels[1],
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, out_channels[2], 3, 2, act=act),
            CSPLayer(
                out_channels[2],
                out_channels[2],
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, out_channels[3], 3, 2, act=act),
            CSPLayer(
                out_channels[3],
                out_channels[3],
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, out_channels[4], 3, 2, act=act),
            SPPBottleneck(out_channels[4],
                          out_channels[4], activation=act),
            CSPLayer(
                out_channels[4],
                out_channels[4],
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

        # used for deconv layers
        if head_convs['hm'][0] == 64:
            # print('Using slimed resnet: 32 32 64 up channels.')
            # self.deconv_layers = self._make_deconv_layer(
            #     3,
            #     [32,32, 64],
            #     [4, 4, 4],
            # )
            print('Using slimed resnet: 256 128 64 up channels.')
            self.deconv_layers = self._make_deconv_layer(
                3,
                [256, 128, 64],
                [4, 4, 4],
            )
        else:
            print('Using original resnet: 256 256 256 up channels.')
            print('Using 256 deconvs')
            self.deconv_layers = self._make_deconv_layer(
                3,
                [256, 256, 256],
                [4, 4, 4],
            )

        self.init_weights()

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes,
                     kernel_size=(3, 3), stride=1,
                     padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1,
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def img2feats(self, x):
        # backbone
        x = self.stem(x) # 2x
        x = self.dark2(x) # 4x
        x = self.dark3(x) # 8x
        x = self.dark4(x) # 16x
        x = self.dark5(x) # 32x

        # DCN
        x = self.deconv_layers(x)

        return [x]

    def img2feats_prev(self, x):
        # backbone
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        x = self.dark4(x)
        x = self.dark5(x)

        # DCN
        x = self.deconv_layers(x)

        # return [x.detach()]
        return [x]
