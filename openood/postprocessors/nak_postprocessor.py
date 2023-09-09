from typing import Any
import gc
import os

from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from .base_postprocessor import BasePostprocessor
from .ekfac import EKFAC
from .info import num_classes_dict
from functorch import make_functional_with_buffers
from torch.func import functional_call, vmap, grad, jacrev, replace_all_batch_norm_modules_
from torch.func import grad

class NAKPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NAKPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.activation_log = None
        self.setup_flag = False

        self.damping = self.args.damping
        self.eigen_iter = self.args.eigen_iter
        self.moments_iter = self.args.moments_iter
        self.left_output = self.args.left_output
        self.right_output = self.args.right_output
        self.top_layer = self.args.top_layer
        self.layer_eps = self.args.layer_eps
        self.eps_type = self.args.eps_type
        self.temperature = self.args.temperature
        self.all_classes = self.args.all_classes
        self.sum_labels = self.args.sum_labels
        self.total_eps = self.args.total_eps
        self.state_path = self.args.state_path
        self.vmap_chunk_size = self.args.vmap_chunk_size
        if self.vmap_chunk_size == -1:
            self.vmap_chunk_size = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        # running stats messes up vmap
        replace_all_batch_norm_modules_(net)

        self.optimizer = EKFAC(
            net,
            eps=self.damping,
            sua=True,
            layer_eps=self.layer_eps,
            eps_type=self.eps_type,
        )

        if not self.setup_flag:

            if self.state_path != "None" and os.path.exists(self.state_path):
                print(f"Loading EKFAC state from {self.state_path}")
                self.optimizer.load_state_dict(torch.load(self.state_path))
            else:
                self.numel = np.sum([p.numel() for p in net.parameters()])
                self.optimizer.init_hooks(net)

                # update over full dataset if not specified
                if self.eigen_iter == -1:
                    # could probably use the last batch but then the equal weighting is thrown off
                    self.eigen_iter = len(id_loader_dict['train']) - 1

                if self.moments_iter == -1:
                    self.moments_iter = len(id_loader_dict['train'])

                # accumulate activations and gradients
                with torch.enable_grad():
                    for i, batch in enumerate(tqdm(id_loader_dict['train'],
                                                   desc="EKFAC State: ",
                                                   position=0,
                                                   leave=True,
                                                   total=self.eigen_iter)):

                        data = batch['data'].cuda()

                        if i == self.eigen_iter:
                            break
                        net.zero_grad()
                        self.optimizer.zero_grad()

                        # sample labels from model distribution
                        logits = net(data)
                        probs = F.softmax(logits / self.temperature, -1)
                        labels = torch.multinomial(probs.detach(), 1).squeeze()
                        loss = F.cross_entropy(logits / self.temperature, labels)
                        loss.backward()
                        self.optimizer.update_state()

                # update eigenbasis
                self.optimizer.average_state(self.eigen_iter)
                self.optimizer.compute_basis()

                # clean up optimizer
                self.optimizer.clear_hooks()
                self.optimizer.clear_cache()
                self.optimizer.zero_grad()
                net.zero_grad()

                # update moments per example
                func, params, buffers = make_functional_with_buffers(net)

                def moments_single(idv_ex, logits):
                    probs = F.softmax(logits / self.temperature, -1)
                    label = F.one_hot(torch.multinomial(probs.detach(), 1).squeeze(), self.num_classes)
                    grads = grad(lambda p, b, d: (-F.log_softmax(func(p, b, d) / self.temperature, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
                    grads = {p: g for p, g in zip(net.parameters(), grads)}
                    return grads

                vmap_moments = vmap(
                    moments_single,
                    in_dims=(0, 0),
                    randomness='different',
                    chunk_size=self.vmap_chunk_size,
                )

                # fit second moments in eigenbasis
                nexamples = 0
                for i, batch in enumerate(tqdm(id_loader_dict['train'],
                                               desc="EKFAC Moments: ",
                                               position=0,
                                               leave=True,
                                               total=self.moments_iter)):
                    if i == self.moments_iter:
                        break
                    nexamples += len(data)
                    data = batch['data'].cuda()
                    logits = net(data)
                    grads = vmap_moments(data, logits)
                    self.optimizer.update_moments(grads=grads)

                self.optimizer.average_moments(nexamples)
                self.optimizer.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()

                if self.state_path != "None":
                    torch.save(self.optimizer.state_dict(), self.state_path)

            self.optimizer.print_eigens()
            self.setup_flag = True

        else:
            pass


    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        data = data.cuda()
        logits, features = net.forward(data, return_feature=True)
        pred = torch.argmax(logits, -1)

        if self.top_layer:
            func, params, buffers = make_functional_with_buffers(net.fc)
        else:
            func, params, buffers = make_functional_with_buffers(net)

        # helper function for vmapping
        def process_single(idv_ex):
            fjac = jacrev(lambda p, b, d: -F.log_softmax(func(p, b, d) / self.temperature, -1))(params, buffers, idv_ex.unsqueeze(0))
            grads = {p: j[0] for p, j in zip(net.parameters(), fjac)}
            nat_grads = self.optimizer.step(grads=grads)
            self_nak = self.optimizer.dict_bdot(grads, nat_grads)
            return self_nak.diagonal()

        vmap_process = vmap(process_single, in_dims=(0,), chunk_size=self.vmap_chunk_size)
        nak = vmap_process(data)
        conf = torch.sum(-nak * F.softmax(logits, -1), -1)

        return pred, conf

