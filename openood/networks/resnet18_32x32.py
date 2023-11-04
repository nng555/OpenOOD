import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

def randomize_bn(bn):
    w_size = bn.weight.shape
    mu_t = torch.rand(w_size) * 0.2 - 0.1 # U[-0.1, 0.1]
    sigma_t = torch.rand(w_size) * 0.25 + 1 # U[1, 1.25]
    bn.bias = nn.Parameter((torch.rand(w_size) * 0.2 - 0.1) + mu_t) # U[mu - 0.1, mu + 0.1]
    bn.weight = nn.Parameter(torch.randn(w_size) * 0.1 + sigma_t)
    bn.weight.requires_grad = False
    bn.bias.requires_grad = False

# fuse conv and bn layer together, assuming no bias
def fuse_conv(conv, bn):
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )
    fused_conv.weight = nn.Parameter(conv.weight * bn.weight[:, None, None, None] / torch.sqrt(bn.running_var + bn.eps)[:, None, None, None])
    fused_conv.bias = nn.Parameter((-bn.running_mean * bn.weight / torch.sqrt(bn.running_var + bn.eps)) + bn.bias)
    return fused_conv

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn=True, bn_affine=True, random_affine=False):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(planes, affine=bn_affine)
            if random_affine:
                randomize_bn(self.bn1)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(planes, affine=bn_affine)
            if random_affine:
                randomize_bn(self.bn2)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if self.use_bn:
                sbn = nn.BatchNorm2d(self.expansion * planes, affine=bn_affine)
                if random_affine:
                    randomize_bn(sbn)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes,
                              self.expansion * planes,
                              kernel_size=1,
                              stride=stride,
                              bias=False),
                    sbn,
                )
            else:
                self.shortcut = nn.Sequential(nn.Conv2d(in_planes,
                                          self.expansion * planes,
                                          kernel_size=1,
                                          stride=stride,
                                          bias=False))

    def fuse_conv_bn_layers(self):
        assert self.use_bn, 'can only fuse if using BN'
        self.use_bn = False
        self.conv1 = fuse_conv(self.conv1, self.bn1)
        self.conv2 = fuse_conv(self.conv2, self.bn2)
        if len(self.shortcut) != 0:
            self.shortcut = fuse_conv(self.shortcut[0], self.shortcut[1])
        del self.bn1
        del self.bn2

    def forward(self, x):
        if self.use_bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)

        sout = out + self.shortcut(x)
        sout = F.relu(sout)
        return sout


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_bn=True, bn_affine=True, random_affine=False):
        super(Bottleneck, self).__init__()
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(planes, affine=bn_affine)
            if random_affine:
                randomize_bn(self.bn1)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(planes, affine=bn_affine)
            if random_affine:
                randomize_bn(self.bn2)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        if self.use_bn:
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=bn_affine)
            if random_affine:
                randomize_bn(self.bn3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if self.use_bn:
                sbn = nn.BatchNorm2d(self.expansion * Planes, affine=bn_affine)
                if random_affine:
                    randomize_bn(self.sbn)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes,
                              self.expansion * planes,
                              kernel_size=1,
                              stride=stride,
                              bias=False),
                    sbn,
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes,
                              self.expansion * planes,
                              kernel_size=1,
                              stride=stride,
                              bias=False))

    def fuse_conv_bn_layers(self):
        assert self.use_bn, 'can only fuse if using BN'
        self.use_bn = False
        self.conv1 = fuse_conv(self.conv1, self.bn1)
        self.conv2 = fuse_conv(self.conv2, self.bn2)
        self.conv3 = fuse_conv(self.conv3, self.bn3)
        if len(self.shortcut) != 0:
            self.shortcut = fuse_conv(self.shortcut[0], self.shortcut[1])
        del self.bn1
        del self.bn2
        del self.bn3

    def forward(self, x):
        if self.use_bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))
            out = self.conv3(out)
        sout = out + self.shortcut(x)
        sout = F.relu(sout)
        return sout


class ResNet18_32x32(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=None, num_classes=10, use_bn=True, bn_affine=True, random_affine=False):
        super(ResNet18_32x32, self).__init__()
        self.use_bn = use_bn
        self.bn_affine = bn_affine
        self.random_affine = random_affine

        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(64, affine=self.bn_affine)
            if self.random_affine:
                randomize_bn(self.bn1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                use_bn=self.use_bn, bn_affine=self.bn_affine, random_affine=self.random_affine)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                use_bn=self.use_bn, bn_affine=self.bn_affine, random_affine=self.random_affine)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                use_bn=self.use_bn, bn_affine=self.bn_affine, random_affine=self.random_affine)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                use_bn=self.use_bn, bn_affine=self.bn_affine, random_affine=self.random_affine)
        # self.avgpool = nn.AvgPool2d(4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.feature_size = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride, use_bn, bn_affine, random_affine):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn, bn_affine, random_affine))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def fuse_conv_bn_layers(self):
        assert self.use_bn, 'can only fuse if using BN'
        print("Fusing Conv and BN layers")
        self.use_bn = False
        self.conv1 = fuse_conv(self.conv1, self.bn1)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.fuse_conv_bn_layers()
        del self.bn1

    def forward(self, x, return_feature=False, return_feature_list=False):
        if self.use_bn:
            feature1 = F.relu(self.bn1(self.conv1(x)))
        else:
            feature1 = F.relu(self.conv1(x))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.fc(feature)
        feature_list = [feature1, feature2, feature3, feature4, feature5]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x, threshold):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.fc(feature)

        return logits_cls

    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        if layer_index == 1:
            return out

        out = self.layer2(out)
        if layer_index == 2:
            return out

        out = self.layer3(out)
        if layer_index == 3:
            return out

        out = self.layer4(out)
        if layer_index == 4:
            return out

        raise ValueError

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc
