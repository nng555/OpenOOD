import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, num_classes, im_size, num_layers, feature_size):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.feature_size = feature_size
        layers = [nn.Linear(im_size, feature_size), nn.ReLU()]
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(feature_size, feature_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(feature_size, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x.view(x.shape[0], -1))
        return out

