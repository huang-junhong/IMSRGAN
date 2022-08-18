import torch.nn as nn
import torch
from torch.nn import functional as F
import pywt
import numpy as np
import torchvision
import random

import functools
import math


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization

        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5 * 0.2 + x

class RRDB(nn.Module):

    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)


    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

class ResUnit(nn.Module):
    def __init__(self, ksize=3, wkdim=64):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(wkdim, wkdim, ksize, 1, int(ksize/2))  
        self.active = nn.PReLU()
        self.conv2 = nn.Conv2d(wkdim, wkdim, ksize, 1, int(ksize/2))

    def forward(self, input):
        current = self.conv1(input)
        current = self.active(current)
        current = self.conv2(current)
        current = input + current
        return current


class UPN(nn.Module):
    def __init__(self, indim=64, scale=2):
        super(UPN, self).__init__()
        self.conv = nn.Conv2d(indim, indim*(scale**2), 3, 1, 1)
        self.Upsample = nn.PixelShuffle(scale)
        self.active = nn.PReLU()

    def forward(self, input):
        current = self.conv(input)
        current = self.Upsample(current)
        current = self.active(current)
        return current

class SRRes(nn.Module):
    def __init__(self, wkdim=64, num_block=16):
        super(SRRes, self).__init__()
        self.head = nn.Conv2d(3, wkdim, 9, 1, 4)
        self.resblock = self._make_resblocks(wkdim, num_block)
        self.gate = nn.Conv2d(wkdim, wkdim, 3, 1, 1)
        self.up_1 = UPN(wkdim)
        self.up_2 = UPN(wkdim)

        self.comp = nn.Conv2d(wkdim*2, wkdim, 3, 1, 1)

        self.tail = nn.Conv2d(wkdim, 3, 9, 1, 4)
            
    def _make_resblocks(self, wkdim, num_block):
        layers = []
        for i in range(1, num_block+1):
            layers.append(ResUnit(wkdim=wkdim))
        return nn.Sequential(*layers)

    def forward(self, input, gate=False):
        F_0 = self.head(input)
        current = self.resblock(F_0)
        current = self.gate(current)
        current = F_0 + current
        if gate:
            return current

        UP_1 = self.up_1(current)
        UP_2 = self.up_2(UP_1)

        current = self.tail(UP_2)
        return current

