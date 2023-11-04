import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, num_classes, num_layers, feature_size):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.feature_size = feature_size
