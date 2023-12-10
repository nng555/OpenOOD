import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class NINBlock(nn.Module):

    def __init__(self, channels, kernel_size, padding, use_bn=False):
        super(NINBlock, self).__init__()
        self.channels = channels

        self.use_bn = use_bn
        self.whiten = False
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size=1, stride=1, padding=0)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(channels[1])
            self.bn2 = nn.BatchNorm2d(channels[2])
            self.bn3 = nn.BatchNorm2d(channels[3])

    def _whiten(self, weights):
        assert not self.use_bn, "can't whiten if already using BN"
        self.whiten = True
        self.bn1 = nn.BatchNorm2d(self.channels[0]).cuda().eval()
        self.bn2 = nn.BatchNorm2d(self.channels[1]).cuda().eval()
        self.bn3 = nn.BatchNorm2d(self.channels[2]).cuda().eval()

        self.bn1.running_mean = weights[0][0][:-1]
        self.bn1.running_var = weights[0][1][:-1]
        self.conv1.bias = nn.Parameter(self.conv1.bias + (self.bn1.running_mean[None, :] * self.conv1.weight.sum(-1).sum(-1)).sum(-1))
        self.conv1.weight = nn.Parameter(self.conv1.weight * torch.sqrt(self.bn1.running_var + self.bn1.eps)[None, :, None, None])

        self.bn2.running_mean = weights[1][0][:-1]
        self.bn2.running_var = weights[1][1][:-1]
        self.conv2.bias = nn.Parameter(self.conv2.bias + (self.bn2.running_mean[None, :] * self.conv2.weight.sum(-1).sum(-1)).sum(-1))
        self.conv2.weight = nn.Parameter(self.conv2.weight * torch.sqrt(self.bn2.running_var + self.bn2.eps)[None, :, None, None])

        self.bn3.running_mean = weights[2][0][:-1]
        self.bn3.running_var = weights[2][1][:-1]
        self.conv3.bias = nn.Parameter(self.conv3.bias + (self.bn3.running_mean[None, :] * self.conv3.weight.sum(-1).sum(-1)).sum(-1))
        self.conv3.weight = nn.Parameter(self.conv3.weight * torch.sqrt(self.bn3.running_var + self.bn3.eps)[None, :, None, None])

    def forward(self, x):
        if self.use_bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = F.relu(self.bn3(self.conv3(out)))
        else:
            if self.whiten:
                out = F.relu(self.conv1(self.bn1(x)))
                out = F.relu(self.conv2(self.bn2(out)))
                out = F.relu(self.conv3(self.bn3(out)))
            else:
                out = F.relu(self.conv1(x))
                out = F.relu(self.conv2(out))
                out = F.relu(self.conv3(out))

        return out


class NIN(nn.Module):

    def _whiten(self, weights):
        assert len(weights) == 9, "Incorrect number of weights provided"
        self.whiten = True
        self.block1._whiten(weights[:3])
        self.block2._whiten(weights[3:6])
        self.block3._whiten(weights[6:])

    def __init__(self, num_classes, use_bn=False):
        super(NIN, self).__init__()
        self.num_classes = num_classes
        self.use_bn = use_bn
        self.whiten = False


        if self.use_bn:
            self.block1 = NINBlock([3, 192, 192, 192], 5, 2, use_bn=use_bn)
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.drop1 = nn.Dropout(0.5)
            self.block2 = NINBlock([192, 192, 192, 192], 5, 2, use_bn=use_bn)
            self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.drop2 = nn.Dropout(0.5)
        else:
            self.block1 = NINBlock([3, 192, 160, 96], 5, 2, use_bn=use_bn)
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.drop1 = nn.Dropout(0.5)
            self.block2 = NINBlock([96, 192, 192, 192], 5, 2, use_bn=use_bn)
            self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.drop2 = nn.Dropout(0.5)
        self.block3 = NINBlock([192, 192, 192, 10], 3, 1, use_bn=use_bn)
        self.pool3 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)

        for block in [self.block1, self.block2, self.block3]:
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.05)
                    m.bias.data.normal_(0, 0.0)

    def forward(self, x):
        out = self.block1(x)
        out = self.drop1(self.pool1(out))
        out = self.block2(out)
        out = self.drop2(self.pool2(out))
        out = self.block3(out)
        out = self.pool3(out)
        out = out.view(x.size(0), self.num_classes)
        return out
