from typing import Any
import time
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

import openood.utils.comm as comm

from openood.preprocessors.transform import normalization_dict
from torch.func import functional_call, vmap, grad, jacrev, replace_all_batch_norm_modules_
from torch.func import grad
from .info import num_classes_dict

class BasePostprocessor:
    def __init__(self, config):
        self.config = config

        #self.num_classes = num_classes_dict[self.config.dataset.name]
        #self.fgsm = self.config.postprocessor.fgsm
        self.fgsm = False

        # FGSM
        """
        try:
            self.input_std = normalization_dict[self.config.dataset.name][1]
            self.input_mean = normalization_dict[self.config.dataset.name][0]
            self.epsilon = self.config.postprocessor.epsilon
        except KeyError:
            self.input_std = [0.5, 0.5, 0.5]
            self.input_mean = [0.5, 0.5, 0.5]


        self.input_std = torch.Tensor(self.input_std).cuda()
        self.input_mean = torch.Tensor(self.input_mean).cuda()
        """

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, labels=None):
        timer = time.time()
        output = net.forward(data)
        ptime = time.time() - timer
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf, {'probs': score.cpu().numpy(), 'ptime': ptime}

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True,
                  use_labels=False):
        pred_list, conf_list, label_list = [], [], []
        extras = defaultdict(list)
        nbatches = len(data_loader)
        for i, batch in enumerate(tqdm(data_loader,
                          disable=not progress or not comm.is_main_process())):
            data = batch['data'].cuda()
            if next(net.parameters()).dtype == torch.float64:
                data = data.double()
            label = batch['label'].cuda()

            if self.fgsm and use_labels:

                net.eval()
                orig_out = net(data)

                params = {k: v.detach() for k, v in net.named_parameters()}
                buffers = {k: v.detach() for k, v in net.named_buffers()}

                def fgsm_fn(idv_ex, target):
                    label = F.one_hot(target, self.num_classes)
                    def loss(d, p, b):
                        logits = functional_call(net, (p, b), (d,))
                        loss = (-F.log_softmax(logits, -1) * label).sum()
                        return loss
                    x_grads = grad(loss)(idv_ex.unsqueeze(0), params, buffers)
                    grad_sign = x_grads.sign()
                    return grad_sign

                grad_sign = vmap(fgsm_fn, in_dims=(0, 0))(data, label).squeeze()

                orig_image = data * self.input_std.view(1, -1, 1, 1) + self.input_mean.view(1, -1, 1, 1)

                success = torch.zeros(len(data)).cuda()

                eps = 0.01
                while eps <= self.epsilon:
                    if (success == 0).sum() == 0:
                        break
                    sidxs = (success == 0).nonzero()[:, 0]
                    fgsm_image = orig_image[sidxs] + eps * grad_sign[sidxs]
                    fgsm_image = torch.clamp(fgsm_image, 0, 1)
                    fgsm_image = (fgsm_image - self.input_mean.view(1, -1, 1, 1)) / self.input_std.view(1, -1, 1, 1)
                    fgsm_out = net(fgsm_image)
                    new_success = sidxs[orig_out[sidxs].argmax(-1) != fgsm_out.argmax(-1)]
                    if len(new_success) == 0:
                        eps += 0.01
                        continue
                    success[new_success] = eps
                    eps += 0.01

                fgsm_image = orig_image + success[:, None, None, None] * grad_sign
                fgsm_image = torch.clamp(fgsm_image, 0, 1)
                data = (fgsm_image - self.input_mean.view(1, -1, 1, 1)) / self.input_std.view(1, -1, 1, 1)

                new_out = net(data)

                success = (new_out.argmax(-1) != orig_out.argmax(-1))
                if success.sum() == 0:
                    continue
                data = data[success]
                label = label[success]

            if use_labels:
                res = self.postprocess(net, data, label)
            else:
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
            if isinstance(extras[k][0], float):
                extras[k] = np.array(extras[k])
            elif isinstance(extras[k][0], list):
                extras[k] = [np.concatenate([ex[i] for ex in extras[k]]) for i in range(len(extras[k][0]))]
            else:
                extras[k] = np.concatenate(extras[k])

        if len(extras) == 0:
            extras = None

        return pred_list, conf_list, label_list, extras
