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
        if self.args.topk == -1:
            self.topk = self.num_classes
        else:
            self.topk = self.args.topk
        self.exemplars = {}
        self.nak_exemplars = {}
        self.avg_grad = None
        self.rand_avg_grad = None
        self.mdl_weight = self.args.mdl_weight
        self.top_layer = self.args.top_layer
        self.layer_eps = self.args.layer_eps
        self.eps_type = self.args.eps_type
        self.temperature = self.args.temperature
        self.relative = self.args.relative
        self.all_classes = self.args.all_classes
        self.sum_labels = self.args.sum_labels
        self.inverse = self.args.inverse
        self.max_examples = self.args.max_examples
        self.normalize = self.args.normalize
        self.sample = self.args.sample
        self.topp = self.args.topp

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
                                               total=self.maxiter)):

                    data = batch['data'].cuda()

                    if i == self.maxiter:
                        break
                    net.zero_grad()
                    self.optimizer.zero_grad()

                    # sample labels from model distribution
                    logits = net(data)
                    probs = F.softmax(logits, -1)
                    labels = torch.multinomial(probs.detach(), 1).squeeze()
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    if self.avg_grad is None:
                        self.avg_grad = {p: p.grad.data / self.maxiter for p in net.parameters()}
                    else:
                        self.avg_grad = {p: p.grad.data / self.maxiter + self.avg_grad[p] for p in net.parameters()}

            # manually compute KFE
            eigens = []
            for group in self.optimizer.param_groups:
                # Getting parameters
                if len(group['params']) == 2:
                    weight, bias = group['params']
                else:
                    weight = group['params'][0]
                    bias = None
                state = self.optimizer.state[weight]
                # Update convariances and inverses
                if group['layer_type'] == 'BatchNorm2d':
                    eigens.append(self.optimizer._update_diagonal(group, state, normalize=self.normalize))
                else:
                    eigens.append(self.optimizer._compute_kfe(group, state, normalize=self.normalize))

            self.optimizer.__del__()

            """
            with torch.enable_grad():
                for i, batch in enumerate(tqdm(id_loader_dict['train'],
                                               desc="Average Grad: ",
                                               position=0,
                                               leave=True,
                                               total=self.maxiter)):
                    if i == 50:
                        break
                    data = batch['data'].cuda()
                    logits = net(data)
                    net.zero_grad()
                    self.optimizer.zero_grad()
                    # maybe sample labels?
                    #labels = torch.multinomial(probs.detach(), 1).squeeze()
                    #logits[labels].mean().backward()
                    #labels = batch['label'].cuda()
                    loss = F.cross_entropy(logits, torch.ones_like(logits))
                    loss.backward()

                    if self.avg_grad is None:
                        self.avg_grad = {p: p.grad.data / self.maxiter for p in net.parameters()}
                    else:
                        self.avg_grad = {p: self.avg_grad[p] + p.grad.data / self.maxiter for p in net.parameters()}

            net.zero_grad()
            self.optimizer.zero_grad()
            self.avg_grad = {p: g.unsqueeze(0) for p, g in self.avg_grad.items()}
            nat_grads = self.optimizer.step(grads=self.avg_grad)
            nak = self.optimizer.dict_dot(self.avg_grad, nat_grads)
            self.avg_grad = {p: g.squeeze(0) / torch.sqrt(nak) for p, g in self.avg_grad.items()}

            #avg_grad_norm = self.optimizer.dict_dot(self.avg_grad, self.avg_grad)
            #self.avg_grad = {p: g / torch.sqrt(avg_grad_norm) for p, g in self.avg_grad.items()}

            self.nak_grads = [None for _ in range(self.num_classes)]
            self.naks = [0 for _ in range(self.num_classes)]
            self.nak_dots = [0 for _ in range(self.num_classes)]
            """

            """
            for k, k_loader in enumerate(id_loader_dict['train_class']):
                print(f"Processing class {k} NAKs")
                nexamples = 0
                for i, batch in enumerate(k_loader):
                    if nexamples > self.max_examples:
                        break
                    nexamples += len(batch['data'])
                    label = batch['label'][0].unsqueeze(0).cuda()
                    assert k == label.item()
                    with torch.enable_grad():
                        for idv_example in batch['data'].cuda():
                            net.zero_grad()
                            self.optimizer.zero_grad()

                            logits = net(idv_example.unsqueeze(0))
                            #loss = F.cross_entropy(logits, logits.argmax().unsqueeze(0))
                            loss = F.cross_entropy(logits, torch.ones_like(logits))
                            loss.backward()
                            #logits[0][k].backward()
                            #logits.max(-1).sum().backward()
                            raw_grads = {p: p.grad.data for p in net.parameters()}
                            self.optimizer.step()
                            nat_grads = {p: p.grad.data for p in net.parameters()}
                            nak = self.optimizer.dict_dot(raw_grads, nat_grads)
                            self.naks[k] += nak
                            #raw_grads = {p: g / torch.sqrt(nak) for p, g in raw_grads.items()}
                            if self.nak_grads[k] is None:
                                self.nak_grads[k] = raw_grads
                            else:
                                self.nak_grads[k] = {p: g + self.nak_grads[k][p] for p, g in raw_grads.items()}

                print(f"Processed {nexamples} examples")
                # average across examples
                net.zero_grad()
                self.optimizer.zero_grad()

                self.nak_grads[k] = {p: v.unsqueeze(0) / nexamples for p, v in self.nak_grads[k].items()}
                nat_grads = self.optimizer.step(grads=self.nak_grads[k])
                total_nak = self.optimizer.dict_dot(self.nak_grads[k], nat_grads)
                self.nak_grads[k] = {p: v.squeeze(0) / torch.sqrt(total_nak) for p, v in self.nak_grads[k].items()}
                self.naks[k] = self.naks[k] / nexamples
                self.nak_dots[k] = self.optimizer.dict_dot(self.nak_grads[k], self.nak_grads[k])
                #self.nak_grads[k] = {p: v / torch.sqrt(self.nak_dots[k]) for p, v in self.nak_grads[k].items()}
            """
            #print([nak.diagonal().sum() for nak in self.naks])
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
                        fjac = jacrev(lambda p, b, d: -F.log_softmax(func(p, b, d), -1))(params, buffers, idv_batch.unsqueeze(0))
                    grads = {p: j[0] for p, j in zip(net.parameters(), fjac)}
                    """
                    # TODO: maybe handle the non-transformed grads?

                nat_grads = self.optimizer.step(grads=grads, inverse=True)
                self_nak = self.optimizer.dict_bdot(grads, nat_grads)

                if self.right_output != 'logits':
                    right_link = self.get_link_fn(self.right_output)
                    rgrad = jacrev(right_link)(logits).squeeze()
                    self_nak = torch.einsum('lr,or -> lo', self_nak, rgrad)

                if self.left_output != 'logits':
                    left_link = self.get_link_fn(self.left_output)
                    lgrad = jacrev(left_link)(logits).squeeze()
                    self_nak = torch.einsum('ol,lr -> or', lgrad, self_nak)

                del grads
                del fjac
                gc.collect()
                torch.cuda.empty_cache()

                if self.sample == -1:
                    if self.topp < 1:
                        probs = F.softmax(logits)
                        sort_probs, sort_idxs = torch.sort(probs, descending=True)
                        cum_probs = torch.cumsum(sort_probs, dim=-1)
                        top_idxs = sort_idxs[:(cum_probs < self.topp).sum() + 1]
                        res = -self_nak.diagonal()[top_idxs] @ probs[top_idxs]
                        res /= probs[top_idxs].sum()
                        conf.append(res)
                    else:
                        conf.append(-self_nak.diagonal() @ F.softmax(logits))
                elif self.sample == 0:
                    conf.append(-self_nak.diagonal().sum())
                else:
                    probs = F.softmax(logits)
                    samples = torch.multinomial(probs.detach(), self.sample, replacement=True).squeeze()
                    res = -self_nak.diagonal()[samples].mean()
                    conf.append(res)

                import ipdb; ipdb.set_trace()
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

class EKFAC(Optimizer):

    def __init__(self, net, eps, niters, sua=False, layer_eps=True, eps_type='mean'):
        """ EKFAC Preconditionner for Linear and Conv2d layers.

        Computes the EKFAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            ra (bool): Computes stats using a running average of averaged gradients
                instead of using a intra minibatch estimate
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter

        """
        self.eps = eps
        self.sua = sua
        self.niters = niters
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self.numel = 0
        self.layer_eps = layer_eps
        self.eps_type = eps_type
        self._iteration_counter = 1
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d', 'BatchNorm2d']:
                params = [mod.weight]
                self.numel += mod.weight.numel()
                if mod.bias is not None:
                    params.append(mod.bias)
                    self.numel += mod.bias.numel()
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                if mod_class == 'Conv2d':
                    if not self.sua:
                        # Adding gathering filter for convolution
                        d['gathering_filter'] = self._get_gathering_filter(mod)
                self.params.append(d)

        super(EKFAC, self).__init__(self.params, {})

    def init_hooks(self, net, avg):
        print("Setting average hooks to " + str(avg))
        self.__del__()
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d', 'BatchNorm2d']:
                if avg:
                    handle = mod.register_forward_pre_hook(self._save_avg_input)
                else:
                    handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                if avg:
                    handle = mod.register_full_backward_hook(self._save_avg_grad_output)
                else:
                    handle = mod.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)

    def step(self, update_stats=True, update_params=True, grads=None, inverse=True, eps=None):
        """Performs one step of preconditioning."""
        nat_grads = {}
        if eps is not None:
            tmp_eps = self.eps
            self.eps = eps

        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            # Update convariances and inverses
            #if self._iteration_counter % self.update_freq == 0:
            #    self._compute_kfe(group, state)
            # Preconditionning
            wgrad = None
            bgrad = None
            if grads is not None:
                wgrad = grads[weight]
                if bias is not None:
                    bgrad = grads[bias]

            if group['layer_type'] == 'Conv2d' and self.sua:
                nat_bgrad, nat_wgrad = self._precond_sua_ra(weight, bias, group, state, wgrad, bgrad, inverse=inverse)
            elif group['layer_type'] == 'BatchNorm2d':
                nat_bgrad, nat_wgrad = self._precond_diag_ra(weight, bias, group, state, wgrad, bgrad, inverse=inverse)
            else:
                nat_bgrad, nat_wgrad = self._precond_ra(weight, bias, group, state, wgrad, bgrad, inverse=inverse)

            if nat_wgrad is not None:
                nat_grads[weight] = nat_wgrad
                if nat_bgrad is not None:
                    nat_grads[bias] = nat_bgrad

        if eps is not None:
            self.eps = tmp_eps

        return nat_grads

    def dict_dot(self, ldict, rdict):
        res = 0
        for p in rdict:
            res += torch.dot(rdict[p].flatten(), ldict[p].flatten()) / self.numel
        return res

    def dict_bdot(self, ldict, rdict):
        res = None
        for p in rdict:
            res_cum = torch.mm(rdict[p].reshape(rdict[p].shape[0], -1), ldict[p].reshape(ldict[p].shape[0], -1).t())
            if res is None:
                res = res_cum / self.numel
            else:
                res += res_cum / self.numel
        return res

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _save_avg_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if 'x' not in self.state[mod]:
            self.state[mod]['x'] = i[0]
        else:
            self.state[mod]['x'] = torch.cat((self.state[mod]['x'], i[0]))

    def _save_avg_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if 'gy' not in self.state[mod]:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)
        else:
            self.state[mod]['gy'] = torch.cat((self.state[mod]['gy'], grad_output[0] * grad_output[0].size(0)))

    def _precond_ra(self, weight, bias, group, state, g=None, gb=None, inverse=True):
        """Applies preconditioning."""
        res = []
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        m2 = state['m2']
        if g is None:
            g = weight.grad.data.unsqueeze(0)
        s = g.shape
        #bs = self.state[group['mod']]['x'].size(0)
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1], s[2]*s[3]*s[4])
        if bias is not None:
            if gb is None:
                gb = bias.grad.data.unsqueeze(0)
            g = torch.cat([g, gb.view(gb.shape[0], gb.shape[1], 1)], dim=2)

        g_kfe = torch.matmul(torch.matmul(kfe_gy.t(), g), kfe_x)

        #m2 = m2.add(g_kfe**2)
        # layerwise multiplicative damping
        eps = self.get_eps(m2)

        if inverse:
            g_nat_kfe = g_kfe / (m2 + eps)
        else:
            g_nat_kfe = g_kfe * (m2 + eps)
        g_nat = torch.matmul(torch.matmul(kfe_gy, g_nat_kfe), kfe_x.t())

        if bias is not None:
            gb = g_nat[:, :, -1].contiguous().view(s[0], *bias.shape)
            if bias.grad is not None:
                bias.grad.data = gb
                res.append(None)
            else:
                res.append(gb)
            g_nat = g_nat[:, :, :-1]
        else:
            res.append(None)

        g_nat = g_nat.contiguous().view(*s)
        if weight.grad is not None:
            weight.grad.data = g_nat
            res.append(None)
        else:
            res.append(g_nat)

        return res

    def _precond_sua_ra(self, weight, bias, group, state, g=None, gb=None, inverse=True):
        """Preconditioning for KFAC SUA."""
        res = []
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        m2 = state['m2']
        if g is None:
            g = weight.grad.data.unsqueeze(0)
        s = g.shape
        #bs = self.state[group['mod']]['x'].size(0)
        mod = group['mod']
        if bias is not None:
            if gb is None:
                gb = bias.grad.data.unsqueeze(0)
            gb = gb.view(s[0], -1, 1, 1, 1).expand(-1, -1, -1, s[3], s[4])
            g = torch.cat([g, gb], dim=2)
        g_kfe = self._to_kfe_sua(g, kfe_x, kfe_gy)
        #m2 = m2.add(g_kfe**2)

        # layerwise multiplicative damping
        eps = self.get_eps(m2)

        if inverse:
            g_nat_kfe = g_kfe / (m2 + eps)
        else:
            g_nat_kfe = g_kfe * (m2 + eps)
        g_nat = self._to_kfe_sua(g_nat_kfe, kfe_x.t(), kfe_gy.t())
        if bias is not None:
            gb = g_nat[:, :, -1, s[3]//2, s[4]//2]
            if bias.grad is not None:
                bias.grad.data = gb.squeeze()
                res.append(None)
            else:
                res.append(gb)
            g_nat = g_nat[:, :, :-1]
        else:
            res.append(None)

        if weight.grad is not None:
            weight.grad.data = g_nat.squeeze()
            res.append(None)
        else:
            res.append(g_nat)

        return res

    def get_eps(self, eigens):
        if self.layer_eps:
            if self.eps_type == 'mean':
                return self.eps * eigens.mean()
            elif self.eps_type == 'max':
                return self.eps * eigens.max()
            elif self.eps_type == 'min':
                if eigens.min() < 0:
                    return (1 + self.eps) * torch.abs(eigens[eigens != 0].min())
                return self.eps * torch.abs(eigens[eigens != 0].min())
        else:
            return self.eps

    def _precond_diag_ra(self, weight, bias, group, state, g=None, gb=None, inverse=True):
        res = []
        if g is None:
            g = weight.grad.data.unsqueeze(0)
        fw = state['exact_fw']
        eps = self.get_eps(fw)

        if inverse:
            g_nat = g / (fw + eps)
        else:
            g_nat = g * (fw + eps)

        if bias is not None:
            if gb is None:
                gb = bias.grad.data.unsqueeze(0)
            fb = state['exact_fb']
            eps = self.get_eps(fb)

            if inverse:
                gb_nat = gb / (fb + eps)
            else:
                gb_nat = gb * (fb + eps)

            if bias.grad is not None:
                bias.grad.data = gb_nat.squeeze()
                res.append(None)
            else:
                res.append(gb_nat)
        else:
            res.append(None)

        if weight.grad is not None:
            weight.grad.data = g_nat.squeeze()
            res.append(None)
        else:
            res.append(g_nat)

        return res

    def _update_diagonal(self, group, state, normalize=False):
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']

        if normalize:
            x -= x.mean()
            gy -= gy.mean()

        # full gradients for shift/scale layer average across batch
        # sum (not mean right??) across spatial dimensions
        gw = torch.sum(x * gy, (-1, -2))
        gw = (gw**2).mean(0)
        if mod.bias is not None:
            gb = torch.sum(gy, (-1, -2))
            gb = (gb**2).mean(0)
            state['exact_fb'] = gb

        state['exact_fw'] = gw
        print(mod)
        print(f"Max Eigenvalue: {gw.max()}, Min Eigenvalue: {gw.min()}, Mean Eigenvalue: {gw.mean()}")
        return torch.cat((gw.flatten(), gb.flatten()))

    def _compute_kfe(self, group, state, normalize=False):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                x = F.conv2d(x, group['gathering_filter'],
                             stride=mod.stride, padding=mod.padding,
                             groups=mod.in_channels)
            x = x.data.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)

        if normalize:
            x -= x.mean()

        xxt = torch.mm(x, x.t()) / float(x.shape[1])
        xxt = xxt.to(torch.device('cuda:0'))
        Ex, state['kfe_x'] = torch.linalg.eigh(xxt.cpu(), UPLO='U')
        Ex = Ex.cuda()
        state['kfe_x'] = state['kfe_x'].cuda()
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1

        if normalize:
            gy -= gy.mean()

        ggt = torch.mm(gy, gy.t()) / float(gy.shape[1])
        ggt = ggt.to(torch.device('cuda:0'))
        Eg, state['kfe_gy'] = torch.linalg.eigh(ggt, UPLO='U')
        Eg = Eg.cuda()
        state['kfe_gy'] = state['kfe_gy'].cuda()
        state['m2'] = Eg.unsqueeze(1) * Ex.unsqueeze(0) * state['num_locations']
        print(mod)
        print(f"Max Eigenvalue: {state['m2'].max()}, Min Eigenvalue: {state['m2'].min()}, Mean Eigenvalue: {state['m2'].mean()}")
        if group['layer_type'] == 'Conv2d' and self.sua:
            ws = group['params'][0].grad.data.size()
            state['m2'] = state['m2'].view(Eg.size(0), Ex.size(0), 1, 1).expand(-1, -1, ws[2], ws[3])
        return state['m2'].flatten()

    def _get_gathering_filter(self, mod):
        """Convolution filter that extracts input patches."""
        kw, kh = mod.kernel_size
        g_filter = mod.weight.data.new(kw * kh * mod.in_channels, 1, kw, kh)
        g_filter.fill_(0)
        for i in range(mod.in_channels):
            for j in range(kw):
                for k in range(kh):
                    g_filter[k + kh*j + kw*kh*i, 0, j, k] = 1
        return g_filter

    def _to_kfe_sua(self, g, vx, vg):
        """Project g to the kfe"""
        sg = g.size()
        g = torch.matmul(vg.t(), g.reshape(sg[0], sg[1], -1)).view(sg[0], vg.size(1), sg[2], sg[3], sg[4])
        g = torch.matmul(g.permute(0, 1, 3, 4, 2).contiguous().view(sg[0], -1, sg[2]), vx)
        g = g.view(sg[0], vg.size(1), sg[3], sg[4], vx.size(1)).permute(0, 1, 4, 2, 3)
        return g

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
        for state in self.state.values():
            if 'x' in state:
                #import ipdb; ipdb.set_trace()
                #state['eg'] = state['x'] @ state['gy']
                del state['x']
                del state['gy']
