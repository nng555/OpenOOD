import ast
import io
import logging
import os

import torch
from PIL import Image, ImageFile
import numpy as np

from .base_dataset import BaseDataset


class NPDataset(BaseDataset):
    def __init__(self,
                 name,
                 data_dir,
                 corruption,
                 label_pth,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 sev=-1,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(NPDataset, self).__init__(**kwargs)

        self.name = name

        self.data_dir = data_dir
        self.np_data = np.load(os.path.join(self.data_dir, corruption + ".npy"))
        self.labels = np.load(os.path.join(self.data_dir, label_pth))

        if sev != -1:
            self.np_data = self.np_data[sev*10000:(sev+1)*10000]
            self.labels = self.labels[sev*10000:(sev+1)*10000]

        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')

    def __len__(self):
        if self.maxlen is None:
            return len(self.np_data)
        else:
            return min(len(self.np_data), self.maxlen)

    def getitem(self, index):
        sample = dict()

        if self.dummy_size is not None:
            sample['data'] = torch.rand(self.dummy_size)
        else:
            image = Image.fromarray(self.np_data[index])
            sample['data'] = self.transform_image(image)
            sample['data_aux'] = self.transform_aux_image(image)

        sample['label'] = self.labels[index]

        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes)
        if sample['label'] < 0:
            soft_label.fill_(1.0 / self.num_classes)
        else:
            soft_label.fill_(0)
            soft_label[sample['label']] = 1
        sample['soft_label'] = soft_label

        return sample
