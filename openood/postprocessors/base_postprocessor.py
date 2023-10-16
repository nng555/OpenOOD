from typing import Any
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
            label = batch['label'].cuda()
            pred, conf, extra = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

            if extra is not None:
                for k, v in extra.items():
                    if k == 'eigenfeat':
                        if len(extras[k]) == 0:
                            extras[k] = v.cpu() / nbatches
                        else:
                            extras[k] += v.cpu() / nbatches
                    else:
                        extras[k].append(v.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        for k in extras:
            if k == 'eigenfeat':
                extras[k] = extras[k].numpy()
            else:
                extras[k] = torch.cat(extras[k]).numpy()

        if len(extras) == 0:
            extras = None

        return pred_list, conf_list, label_list, extras
