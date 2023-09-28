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
        self.loss_temp = self.args.loss_temp
        self.sample_temp = self.args.sample_temp
        self.fsample_temp = self.args.fsample_temp
        self.floss_temp = self.args.floss_temp
        self.all_classes = self.args.all_classes
        self.sum_labels = self.args.sum_labels
        self.total_eps = self.args.total_eps
        self.state_path = self.args.state_path + '.cpt'
        self.grad_path = self.args.state_path + '_grad.cpt'
        self.jac_chunk_size = self.args.jac_chunk_size
        self.empirical = self.args.empirical
        self.wrt_logits = self.args.wrt_logits
        self.moments_chunk_size = self.args.moments_chunk_size
        self.topk = self.args.topk
        if self.jac_chunk_size == -1:
            self.jac_chunk_size = None
        if self.moments_chunk_size == -1:
            self.moments_chunk_size = None

    # helper function for normalizing gradients
    def param_dot(self, params):
        res = 0
        for k in params:
            res += torch.dot(params[k].flatten(), self.part_grad[k].flatten())
        return res

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        self.optimizer = EKFAC(
            net,
            eps=self.damping,
            sua=True,
            layer_eps=self.layer_eps,
            eps_type=self.eps_type,
        )

        if not self.setup_flag:
            net.eval()

            # running stats messes up vmap
            if self.state_path != "None.cpt" and os.path.exists(self.state_path):
                assert os.path.exists(self.grad_path), "No associated grad path found!"
                print(f"Loading grad state from {self.grad_path}")
                self.part_grad = torch.load(self.grad_path)
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
                    ntrain = 0
                    for i, batch in enumerate(tqdm(id_loader_dict['train'],
                                                   desc="Logit Partition: ",
                                                   position=0,
                                                   leave=True,
                                                   total=self.eigen_iter)):
                        data = batch['data'].cuda()
                        labels = batch['labels'].cuda()
                        ntrain += len(data)

                        if i == self.eigen_iter:
                            break

                        logits = net(data)
                        loss = F.cross_entropy(logits, labels) / self.eigen_iter
                        loss.backward()

                    self.ntrain = ntrain
                    self.part_grad = {v: grads[k] for k, v in net.named_parameters()}

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

                        logits = net(data)
                        if self.fsample_temp == -1:
                            probs = F.softmax(torch.ones_like(logits), -1)
                        else:
                            probs = F.softmax(logits / self.fsample_temp, -1).detach()
                        if self.topk == -1:
                            # sample a single class
                            if self.empirical:
                                labels = batch['label'].cuda()
                            else:
                                labels = torch.multinomial(probs.detach(), 1).squeeze()
                            if self.wrt_logits:
                                loss = -torch.take_along_dim(logits, labels[:, None].long(), -1).mean()
                            else:
                                loss = F.cross_entropy(logits / self.floss_temp, labels)

                            loss.backward()
                            self.optimizer.update_state()
                        else:
                            # manually weight topk classes rather than sampling
                            weights, labels = torch.topk(probs.detach(), self.topk, -1)
                            weights /= weights.sum(-1)[:, None]
                            for kcls in range(self.topk):
                                net.zero_grad()
                                self.optimizer.zero_grad()
                                if self.wrt_logits:
                                    loss = -torch.take_along_dim(logits, labels[:, kcls][:, None].long(), -1).mean()
                                else:
                                    loss = F.cross_entropy(logits / self.floss_temp, labels[:, kcls])
                                loss.backward(retain_graph=(kcls != self.topk - 1))
                                self.optimizer.update_state(weights[:, kcls])

                # update eigenbasis
                self.optimizer.average_state(self.eigen_iter)
                self.optimizer.compute_basis()

                # clean up optimizer
                self.optimizer.clear_hooks()
                self.optimizer.clear_cache()
                self.optimizer.zero_grad()
                net.zero_grad()

                # update moments per example
                params = {k: v.detach() for k, v in net.named_parameters()}
                buffers = {k: v.detach() for k, v in net.named_buffers()}

                def logits_loss(params, buffers, data, label):
                    loss = (-functional_call(net, (params, buffers), (data,)) * label).sum()
                    return loss

                def moments_single_logits(idv_ex, target):
                    label = F.one_hot(target, self.num_classes)
                    grads = grad(logits_loss)(params, buffers, idv_ex.unsqueeze(0), label)
                    return grads

                def moments_single_loss(idv_ex, target):
                    label = F.one_hot(target, self.num_classes)
                    grads = grad(lambda p, b, d: (-F.log_softmax(functional_call(net, (p, b), (d,)) / self.floss_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
                    return grads

                if self.wrt_logits:
                    grad_fn = moments_single_logits
                else:
                    grad_fn = moments_single_loss

                vmap_moments = vmap(
                    grad_fn,
                    in_dims=(0, 0),
                    randomness='different',
                    chunk_size=self.moments_chunk_size,
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
                    if self.fsample_temp == -1:
                        probs = F.softmax(torch.ones_like(logits), -1)
                    else:
                        probs = F.softmax(logits / self.fsample_temp, -1).detach()
                    if self.topk == -1:
                        # sample a label
                        if self.empirical:
                            targets = batch['label'].cuda()
                        else:
                            targets = torch.multinomial(probs.detach(), 1).squeeze()
                        grads = vmap_moments(data, targets)
                        grads = {v: grads[k] for k, v in net.named_parameters()}
                        self.optimizer.update_moments(grads)
                    else:
                        # manually average over topk classes
                        weights, idxs = torch.topk(probs.detach(), self.topk, -1)
                        weights /= weights.sum(-1)[:, None]
                        for kcls in range(self.topk):
                            grads = vmap_moments(data, idxs[:, kcls])
                            grads = {v: grads[k] for k, v in net.named_parameters()}
                            self.optimizer.update_moments(grads=grads, ex_weight=weights[:, kcls])
                            del grads

                self.optimizer.average_moments(nexamples)
                self.optimizer.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()
                net.zero_grad()
                self.optimizer.zero_grad()

                if self.state_path != "None.cpt":
                    torch.save(self.optimizer.state_dict(), self.state_path)
                    torch.save(self.part_grad, self.grad_path)

            self.optimizer.print_eigens()
            self.setup_flag = True

        else:
            pass


    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        data = data.cuda()
        logits, features = net.forward(data, return_feature=True)
        pred = torch.argmax(logits, -1)

        params = {k: v.detach() for k, v in net.named_parameters()}
        buffers = {k: v.detach() for k, v in net.named_buffers()}

        # helper function for vmapping
        def grad_single_logits(idv_ex, target):
            label = F.one_hot(target, self.num_classes)
            grads = grad(lambda p, b, d: (-functional_call(net, (p, b), (d,)) * label).sum() / self.ntrain)(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: grads[k].unsqueeze(0) for k, v in net.named_parameters()}
            nat_grads = self.optimizer.step(grads=grads)
            self_nak = self.optimizer.dict_dot(grads, nat_grads)
            part_nak = self.optimizer.dict_dot(self.part_grad, nat_grads)
            return self_nak + 2 * part_nak

        def grad_single_loss(idv_ex, target):
            label = F.one_hot(target, self.num_classes)
            grads = grad(lambda p, b, d: (-F.log_softmax(functional_call(net, (p, b), (d,)) / self.loss_temp, -1) * label).sum() / self.ntrain)(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: grads[k].unsqueeze(0) for k, v in net.named_parameters()}
            nat_grads = self.optimizer.step(grads=grads)
            self_nak = self.optimizer.dict_dot(grads, nat_grads)
            part_nak = self.optimizer.dict_dot(self.part_grad, nat_grads)
            return self_nak + 2 * part_nak

        if self.wrt_logits:
            grad_fn = grad_single_logits
        else:
            grad_fn = grad_single_loss

        def process_single_grad(idv_ex):
            return vmap(grad_fn, in_dims=(None, 0))(idv_ex, torch.arange(self.num_classes).cuda())

        def process_single_jac(idv_ex):
            fjac = jacrev(lambda p, b, d: -F.log_softmax(functional_call(net, (p, b), (d,)) / self.loss_temp, -1) / self.ntrain)(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: fjac[k][0] for k, v in net.named_parameters()}
            nat_grads = self.optimizer.step(grads=grads)
            self_nak = self.optimizer.dict_bdot(grads, nat_grads)
            # TODO add partition NAK
            return self_nak.diagonal()

        vmap_jac = vmap(process_single_jac, in_dims=(0,), chunk_size=self.jac_chunk_size)
        vmap_grad = vmap(process_single_grad, in_dims=(0,), chunk_size=self.jac_chunk_size)

        nak = vmap_grad(data)
        #nak = vmap_jac(data)
        if self.sample_temp == -1:
            conf = torch.mean(-nak, -1)
        else:
            conf = torch.sum(-nak * F.softmax(logits / self.sample_temp, -1), -1)

        return pred, conf

