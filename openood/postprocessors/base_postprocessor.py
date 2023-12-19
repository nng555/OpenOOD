from typing import Any
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

import openood.utils.comm as comm


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf, None

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        extras = defaultdict(list)
        nbatches = len(data_loader)
        for i, batch in enumerate(tqdm(data_loader,
                          disable=not progress or not comm.is_main_process())):
            data = batch['data'].cuda()
            if next(net.parameters()).dtype == torch.float64:
                data = data.double()
            label = batch['label'].cuda()

            res = self.postprocess(net, data)
            if len(res) == 2:
                pred, conf = res
                extra = None
            else:
                pred, conf, extra = res

            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()
            pred_list.append(pred)
            if torch.is_tensor(conf):
                conf = conf.cpu().numpy()
            conf_list.append(conf)
            if torch.is_tensor(label):
                label = label.cpu().numpy()
            label_list.append(label)

            if extra is not None:
                for k, v in extra.items():
                    extras[k].append(v)

        # convert values into numpy array
        pred_list = np.concatenate(pred_list).astype(int)
        conf_list = np.concatenate(conf_list)
        label_list = np.concatenate(label_list).astype(int)

        for k in extras:
            if isinstance(extras[k][0], list):
                extras[k] = [np.concatenate([ex[i] for ex in extras[k]]) for i in range(len(extras[k][0]))]
            else:
                extras[k] = np.concatenate(extras[k])

        if len(extras) == 0:
            extras = None

        return pred_list, conf_list, label_list, extras
