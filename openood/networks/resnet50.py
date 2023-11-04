from torchvision.models.resnet import Bottleneck, ResNet


class ResNet50(ResNet):
    def __init__(self,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3],
                 num_classes=1000,
                 cifar=False):
        super(ResNet50, self).__init__(block=block,
                                       layers=layers,
                                       num_classes=num_classes)
        self.feature_size = 2048
        self.cifar = cifar
        if self.cifar:
            import torch.nn as nn
            self.conv1 = nn.Conv2d(3,
                                   64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)

        # make relus not inplace so we can hook through
        self.relu = nn.ReLU(inplace=False)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if hasattr(block, 'relu'):
                    block.relu = nn.ReLU(inplace=False)

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        if not self.cifar:
            feature1 = self.maxpool(feature1)
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
        feature1 = self.relu(self.bn1(self.conv1(x)))
        if not self.cifar:
            feature1 = self.maxpool(feature1)
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
        out = self.relu(self.bn1(self.conv1(x)))
        if not self.cifar:
            out = self.maxpool(out)

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
