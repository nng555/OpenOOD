from typing import Any
import gc
import os

from copy import deepcopy
import torch
torch.cuda.set_device('cuda:0')
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict
from functorch import vmap, jacrev, make_functional_with_buffers
from torch.func import grad

class NAKPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NAKPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.activation_log = None
        self.setup_flag = False

        self.damping = self.args.damping
        self.maxiter = self.args.maxiter
        self.left_output = self.args.left_output
        self.right_output = self.args.right_output
        self.strat = self.args.strat
        self.top_layer = self.args.top_layer
        self.layer_eps = self.args.layer_eps
        self.eps_type = self.args.eps_type
        self.temperature = self.args.temperature
        self.all_classes = self.args.all_classes
        self.sum_labels = self.args.sum_labels
        self.total_eps = self.args.total_eps

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        if not self.top_layer:
            fc = None
            self.optimizer = EKFAC(
                net,
                eps=self.damping,
                niters=self.maxiter,
                sua=True,
                layer_eps=self.layer_eps,
                eps_type=self.eps_type,
            )
        else:
            fc = net.fc
            self.optimizer = EKFAC(
                fc,
                eps=self.damping,
                niters=self.maxiter,
                sua=True,
                layer_eps=self.layer_eps,
                eps_type=self.eps_type,
            )

        if not self.setup_flag:

            if self.top_layer:
                self.numel = np.sum([p.numel() for p in fc.parameters()])
                self.optimizer.init_hooks(fc, True)
            else:
                self.numel = np.sum([p.numel() for p in net.parameters()])
                self.optimizer.init_hooks(net, True)

            if self.mdl_weight == 0:
                self.mdl_weight = len(id_loader_dict['train'].dataset)

            # accumulate activations and gradients
            with torch.enable_grad():
                for i, batch in enumerate(tqdm(id_loader_dict['train'],
                                               desc="EKFAC: ",
                                               position=0,
                                               leave=True,
                                               total=self.eigen_iter)):

                    data = batch['data'].cuda()

                    if i == self.maxiter:
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
            self.optimizer.average_state(self.maxiter)
            self.optimizer.compute_basis()
            self.optimizer.__del__()

            # fit second moments in eigenbasis
            nexamples = 0
            with torch.enable_grad():
                for i, batch in enumerate(tqdm(id_loader_dict['train'],
                                               desc="EKFAC: ",
                                               position=0,
                                               leave=True,
                                               total=self.moment_iter)):
                    nexamples += len(data)
                    data = batch['data'].cuda()
                    for idv_data in data:
                        logits = net(data.unsqueeze(0))
                        probs = F.softmax(logits / self.temperature, -1)
                        labels = torch.multinomial(probs.detach(), 1).squeeze()
                        loss = F.cross_entropy(logits / self.temperature, labels)
                        loss.backward()
                        self.optimizer.update_moments()

            self.optimizer.average_moments(nexamples)

            """
            eigens = torch.cat(eigens)
            print(f"Max Eigenvalue: {eigens.max()}, Min Eigenvalue: {eigens.min()}, Mean Eigenvalue: {eigens.mean()}")
            if self.total_eps:
                assert not self.layer_eps, "Can't set both layer and total damping"
                min_nonzero_ev = torch.min(eigens[eigens != 0])
                if min_nonzero_ev < 0:
                    self.optimizer.eps = (1 + self.optimizer.eps) * torch.abs(min_nonzero_ev)
                else:
                    self.optimizer.eps = self.optimizer.eps * min_nonzero_ev
                print(f"Setting new damping value to {self.optimizer.eps}")

            if self.eps_type == 'replace':
                nreplace = (eigens < self.optimizer.eps).sum() / len(eigens)
                print(f"Replacing {torch.round(nreplace * 100, decimals=2)}% of eigenvalues with {self.optimizer.eps}")
            elif self.eps_type == 'abs_replace':
                nabs = (eigens < 0).sum() / len(eigens)
                nreplace = (eigens == 0).sum() / len(eigens)
                print(f"{torch.round(nabs * 100, decimals=2)}% of eigenvalues turned positive, {torch.round(nreplace * 100, decimals=2)}% of eigenvalues replaced with {self.optimizer.eps}")
            """

            self.setup_flag = True

        else:
            pass

    def loss(self, logits):
        return -F.log_softmax(logits / self.temperature, dim=-1)

    def softmax(self, logits):
        return F.softmax(logits, dim=-1)

    def identity(self, logits):
        return logits

    def energy(self, logits):
        return torch.exp(logits)

    def regret(self, logits):
        return logits + self.energy(logits)

    def brier(self, logits):
        return ((torch.eye(logits.shape[0]).to(logits.device) - F.softmax(logits, dim=-1))**2).sum(-1)

    def softlogits(self, logits):
        probs = F.softmax(logits, -1).detach()
        return probs * logits

    def softloss(self, logits):
        probs = F.softmax(logits, -1).detach()
        return -F.log_softmax(logits / self.temperature, dim=-1) * probs

    def get_link_fn(self, lname):
        if lname == 'loss':
            return self.loss
        elif lname == 'softmax':
            return self.softmax
        elif lname == 'softlogits':
            return self.softlogits
        elif lname == 'softloss':
            return self.softloss
        elif lname == 'logits':
            return self.identity
        elif lname == 'energy':
            return self.energy
        elif lname == 'regret':
            return self.regret
        elif lname == 'brier':
            return self.brier
        else:
            raise NotImplementedError

    def backward_link(self, link, logits, retain_graph=False, sum_labels=False):
        if link == 'loss':
            if sum_labels:
                F.cross_entropy(logits, torch.ones_like(logits) / self.num_classes).backward(retain_graph=retain_graph)
            else:
                (-F.log_softmax(logits, -1))[0][logits.argmax()].backward(retain_graph=retain_graph)
        elif link == 'softmax':
            if sum_labels:
                F.softmax(logits, -1).sum().backward(retain_graph=retain_graph)
            else:
                F.softmax(logits, -1)[0][logits.argmax()].backward(retain_graph=retain_graph)
        elif link == 'logits':
            if sum_labels:
                torch.sum(logits).backward(retain_graph=retain_graph)
            else:
                logits[0][logits.argmax()].backward(retain_graph=retain_graph)
        elif link == 'softlogits':
            probs = F.softmax(logits, -1).detach()
            if sum_labels:
                (logits * probs).sum().backward(retain_graph=retain_graph)
            else:
                (logits[0][logits.argmax()] * probs[0][logits.argmax()]).backward(retain_graph=retain_graph)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        conf = []

        data = data.cuda()
        logits, features = net.forward(data, return_feature=True)
        pred = torch.argmax(logits, -1)

        if not self.all_classes:
            for idv_batch in data:
                with torch.enable_grad():
                    net.zero_grad()
                    self.optimizer.zero_grad()

                    logits = net.forward(idv_batch.unsqueeze(0))
                    label = logits[0].detach().argmax(-1).item()

                    net.zero_grad()
                    self.optimizer.zero_grad()
                    self.backward_link(self.right_output, logits, retain_graph=True, sum_labels=True)

                    right_grads = {p: p.grad.data for p in net.parameters()}
                    if self.normalize:
                        right_grads = {p: g - self.avg_grad[p] for p, g in right_grads.items()}
                    self.optimizer.step(inverse=self.inverse)
                    right_nat_grads = {p: p.grad.data for p in net.parameters()}

                    net.zero_grad()
                    self.optimizer.zero_grad()

                    if self.right_output == self.left_output:
                        left_grads = right_grads
                    else:
                        self.backward_link(self.left_output, logits, retain_graph=True, sum_labels=False)
                        left_grads = {p: p.grad.data for p in net.parameters()}
                        left_grads = {p: g - self.avg_grad[p] for p, g in left_grads.items()}
                    sum_nak = self.optimizer.dict_dot(left_grads, right_nat_grads)

                    if not self.relative:
                        conf.append(-sum_nak)
                        continue

                    net.zero_grad()
                    self.optimizer.zero_grad()
                    self.backward_link(self.right_output, logits, retain_graph=True, sum_labels=False)

                    right_grads = {p: p.grad.data for p in net.parameters()}
                    self.optimizer.step(inverse=self.inverse)
                    right_nat_grads = {p: p.grad.data for p in net.parameters()}

                    net.zero_grad()
                    self.optimizer.zero_grad()

                    if self.right_output == self.left_output:
                        left_grads = right_grads
                    else:
                        self.backward_link(self.left_output, logits, retain_graph=False, sum_labels=False)
                        left_grads = {p: p.grad.data for p in net.parameters()}

                    max_nak = self.optimizer.dict_dot(left_grads, right_nat_grads)

                    conf.append(-max_nak / sum_nak)
        else:

            if self.top_layer:
                func, params, buffers = make_functional_with_buffers(net.fc)
            else:
                func, params, buffers = make_functional_with_buffers(net)

            for logits, idv_batch, idv_feature in zip(logits, data, features):
                net.zero_grad()
                self.optimizer.zero_grad()

                with torch.enable_grad():
                    """
                    if self.top_layer:
                        fjac = jacrev(func)(params, buffers, idv_feature.unsqueeze(0))
                        grads = {p: j[0] for p, j in zip(net.fc.parameters(), fjac)}
                    else:
                        fjac = jacrev(func)(params, buffers, idv_batch.unsqueeze(0))
                        grads = {p: j[0] for p, j in zip(net.parameters(), fjac)}
                    """
                    if self.right_output == 'logits':
                        fjac = jacrev(func)(params, buffers, idv_batch.unsqueeze(0))
                    elif self.right_output == 'softmax':
                        fjac = jacrev(lambda p, b, d: F.softmax(func(p, b, d), -1))(params, buffers, idv_batch.unsqueeze(0))
                    elif self.right_output == 'loss':
                        fjac = jacrev(lambda p, b, d: -F.log_softmax(func(p, b, d) / self.temperature, -1))(params, buffers, idv_batch.unsqueeze(0))
                    grads = {p: j[0] for p, j in zip(net.parameters(), fjac)}

                    # TODO: maybe handle the non-transformed grads?

                if self.normalize:
                    grads = {p: j - self.avg_grad[p] for p, j in grads.items()}

                nat_grads = self.optimizer.step(grads=grads, inverse=True)
                self_nak = self.optimizer.dict_bdot(grads, nat_grads)
                grad_dots = self.optimizer.dict_bdot(grads, grads)

                """
                if self.right_output != 'logits':
                    right_link = self.get_link_fn(self.right_output)
                    rgrad = jacrev(right_link)(logits).squeeze()
                    self_nak = torch.einsum('lr,or -> lo', self_nak, rgrad)

                if self.left_output != 'logits':
                    left_link = self.get_link_fn(self.left_output)
                    lgrad = jacrev(left_link)(logits).squeeze()
                    self_nak = torch.einsum('ol,lr -> or', lgrad, self_nak)
                """

                del grads
                del fjac
                gc.collect()
                torch.cuda.empty_cache()

                if self.sample == -1:
                    if self.topp < 1:
                        probs = F.softmax(logits / self.sample_temperature)
                        sort_probs, sort_idxs = torch.sort(probs, descending=True)
                        cum_probs = torch.cumsum(sort_probs, dim=-1)
                        top_idxs = sort_idxs[:(cum_probs < self.topp).sum() + 1]
                        res = -self_nak.diagonal()[top_idxs] @ probs[top_idxs]
                        res /= probs[top_idxs].sum()
                        conf.append(res)
                    else:
                        conf.append(-self_nak.diagonal() @ F.softmax(logits / self.sample_temperature))
                elif self.sample == 0:
                    if self.topp < 1:
                        probs = F.softmax(logits / self.sample_temperature)
                        sort_probs, sort_idxs = torch.sort(probs, descending=True)
                        cum_probs = torch.cumsum(sort_probs, dim=-1)
                        top_idxs = sort_idxs[:(cum_probs < self.topp).sum() + 1]
                        res = -self_nak.diagonal()[top_idxs].mean()
                        conf.append(res)
                    else:
                        conf.append(-self_nak.diagonal().mean())
                else:
                    probs = F.softmax(logits / self.sample_temperature)
                    samples = torch.multinomial(probs.detach(), self.sample, replacement=True).squeeze()
                    res = -self_nak.diagonal()[samples].mean()
                    conf.append(res)

                """
                if self.relative:
                    conf.append(-self_nak.diagonal()[logits.argmax()] / self_nak.diagonal().sum())
                else:
                    conf.append(-self_nak.diagonal().sum())
                """

                def mdl(weight):
                    if self.left_output == 'softmax':
                        if weight == None:
                            time = (1 - F.softmax(logits)) / self_nak.diagonal()
                            mle_probs = F.softmax(logits) - self_nak.diagonal() * time.min()
                        else:
                            mle_probs = F.softmax(logits) - self_nak.diagonal() / weight
                    elif self.left_output == 'logits':
                        mle_probs = F.softmax(logits - self_nak.t() / weight, -1).diagonal()
                    mdl_probs = mle_probs / mle_probs.sum()
                    return mdl_probs, mle_probs, torch.log(mle_probs.sum())

        conf = torch.stack(conf).cpu()
        return torch.Tensor(pred), torch.Tensor(conf)

