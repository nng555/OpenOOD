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
        self.mdl_weight = self.args.mdl_weight
        self.top_layer = self.args.top_layer
        self.layer_eps = self.args.layer_eps
        self.temperature = self.args.temperature

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        if not self.top_layer:
            fc = None
            self.optimizer = EKFAC(
                net,
                eps=self.damping,
                niters=self.maxiter,
                sua=True,
                layer_eps=self.layer_eps,
            )
        else:
            fc = net.fc
            self.optimizer = EKFAC(
                fc,
                eps=self.damping,
                niters=self.maxiter,
                sua=True,
                layer_eps=self.layer_eps,
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

            # manually compute KFE
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
                    self.optimizer._update_diagonal(group, state)
                else:
                    self.optimizer._compute_kfe(group, state)

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
                    loss = F.cross_entropy(logits, batch['label'].cuda())
                    loss.backward()

                    if self.avg_grad is None:
                        self.avg_grad = {p: p.grad.data / self.maxiter for p in net.parameters()}
                    else:
                        self.avg_grad = {p: self.avg_grad[p] + p.grad.data / self.maxiter for p in net.parameters()}
            self.avg_grad = {p: self.avg_grad[p].unsqueeze(0) for p in self.avg_grad}
            """

            """
            self.nak_grads = []
            self.naks = []
            for k, k_loader in enumerate(id_loader_dict['val_class']):
                net.zero_grad()
                self.optimizer.zero_grad()
                for batch in k_loader:
                    assert k == batch['label'][0].item()
                    with torch.enable_grad():
                        logits = net(batch['data'].cuda())
                        logits[0][k].backward()
                    break
                grads = {p: p.grad.data for p in net.parameters()}
                self.nak_grads.append(grads)
                #self.exemplars[k] = grads
                self.optimizer.step()
                nat_grads = {p: p.grad.data for p in net.parameters()}
                nak = np.sum([(torch.dot(lg.flatten(), rg.flatten()) / self.numel).cpu().numpy() for lg, rg in zip(grads.values(), nat_grads.values())])
                self.naks.append(nak)

            self.nak_grads = {p: torch.stack([grads[p] for grads in self.nak_grads]) for p in self.nak_grads[0]}
            self.naks = torch.Tensor(self.naks).cuda()
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
        return -torch.logsumexp(logits, dim=-1)

    def regret(self, logits):
        return logits + self.energy(logits)

    def brier(self, logits):
        return ((torch.eye(logits.shape[0]).to(logits.device) - F.softmax(logits, dim=-1))**2).sum(-1)

    def get_link_fn(self, lname):
        if lname == 'loss':
            return self.loss
        elif lname == 'softmax':
            return self.softmax
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

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """
        pred = []
        conf = []

        data = data.cuda()
        logits = net.forward(data)
        pred = torch.argmax(logits, -1)

        for logits, idv_batch in zip(logits, data):
            net.zero_grad()
            self.optimizer.zero_grad()

            with torch.enable_grad():
                logits = net.forward(idv_batch.unsqueeze(0))
                label = logits.detach().argmax(-1)
                loss = F.cross_entropy(logits, label)
                loss.backward(retain_graph=True)
                #logits[0][label.item()].backward()
                self.optimizer.step()
                nat_grads = [param.grad.data for param in net.parameters()]
                net.zero_grad()
                self.optimizer.zero_grad()
                loss = F.kl_div(logits, torch.ones_like(logits) / self.num_classes, reduction='batchmean')
                loss.backward()
                raw_grads = [param.grad.data for param in net.parameters()]

                #net.zero_grad()
                #self.optimizer.zero_grad()
                #logits[0][label].backward()

                #ex_grads = [param.grad.data for param in net.parameters()]

            #exemplar = self.exemplars[label.item()]
            #ex_logits = net.forward(exemplar.unsqueeze(0).cuda())
            #ex_loss = F.cross_entropy(ex_logits, label)
            #ex_loss.backward()
            #ex_grads = [param.grad.data for param in net.parameters()]
            #ex_grads = self.nak_grads[label.item()]
            #ex_grads = self.avg_grad

            #self_ex_nak = np.sum([(torch.dot(lg.flatten(), rg.flatten()) / self.numel).cpu().numpy() for lg, rg in zip(ex_grads, nat_grads)])
            self_nak = np.sum([(torch.dot(lg.flatten(), rg.flatten()) / self.numel).cpu().numpy() for lg, rg in zip(raw_grads, nat_grads)])
            conf.append(self_nak)
            #ex_nak = self.naks[label.item()]

            #cosine_sim = self_ex_nak / np.sqrt(self_nak * ex_nak)
            #conf.append(cosine_sim)

            #conf.append(-self_ex_nak + 0.5 * (self_nak + ex_nak))
        """
        pred = []
        conf = []

        data = data.cuda()
        logits, features = net.forward(data, return_feature=True)
        pred = torch.argmax(logits, -1)

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
                # TODO: maybe handle the non-transformed grads?

            self_nak = self.optimizer.step(grads=grads)
            #self_ex_nak = self.optimizer.step(grads=grads, left_grads=self.nak_grads)

            del grads
            del fjac
            gc.collect()
            torch.cuda.empty_cache()

            if self.right_output != 'logits':
                right_link = self.get_link_fn(self.right_output)
                rgrad = jacrev(right_link)(logits).squeeze()
                self_nak = torch.einsum('lr,or -> lo', self_nak, rgrad)
                #self_ex_nak = self_ex_nak @ rgrad.t()

            if self.left_output != 'logits':
                left_link = self.get_link_fn(self.left_output)
                lgrad = jacrev(left_link)(logits).squeeze()
                self_nak = torch.einsum('ol,lr -> or', lgrad, self_nak)
                #self_nak = lgrad @ self_nak

            # output dim is the first dimension if it exists
            #nak_dist = -self_ex_nak.diagonal() + 0.5 * (self_nak.diagonal() + self.naks)
            #nak_dist = -self_ex_nak + 0.5 * self_nak.diagonal()

            #conf.append(nak_dist.min())
            label = logits.detach().argmax(-1)
            probs = F.softmax(logits, -1)
            #conf.append(nak_dist.max())
            #conf.append(self_nak.diagonal().mean())
            conf.append(-self_nak.diagonal().mean())

            #conf.append(-self_nak.diagonal()[label] + probs[label] * self_nak.diagonal().sum())

            #conf.append(nak.diagonal()[label.item()] / nak.diagonal().sum())

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
            #conf.append(self_nak.diagonal() @ F.softmax(logits, -1))
            #conf.append((self_nak @ F.softmax(logits, -1)).sum())

            #mdl_probs, mle_probs, regret = mdl(self.mdl_weight)
            #conf.append(-regret)
            #conf.append(self_nak.sum(0).min() / self_nak.sum())


            #mdl_grad = jacrev(mdl)(torch.tensor(float(self.mdl_weight)))
            #regret = torch.log(mdl) - torch.log(F.softmax(logits))
            #mdl_regret = torch.log(mle_probs) - torch.log(mdl)
            #print(mdl.max() - F.softmax(logits).max())
            #mdl_probs, regret = mdl(self.mdl_weight)
            #time = (1 - F.softmax(logits)) / nak.diagonal()
            #conf.append(((1 - F.softmax(logits)) / nak.diagonal())[label.item()])
            #conf.append(regret.mean())
            #if F.softmax(logits).max() < 0.8:
            #    import ipdb; ipdb.set_trace()
        """
            if self.strat == 'mean':
                nak = nak.diagonal()
                conf.append(nak.mean())
            elif self.strat == 'max':
                nak = nak.diagonal()
                conf.append(nak.max())
            elif self.strat == 'min':
                nak = nak.diagonal()
                conf.append(nak.min())
            elif self.strat == 'soft':
                nak = nak.diagonal()
                conf.append(nak @ F.softmax(logits))
            elif self.strat == 'minmax':
                conf.append(-nak.max(-2)[0].min(-1)[0])
        """

        npconf = torch.Tensor(conf).numpy()
        #print([npconf.max(), npconf.min(), npconf.mean()])
        return torch.Tensor(pred), torch.Tensor(conf)

        """
            with torch.enable_grad():
                # maybe diff strategy here...
                nak = []
                for k in torch.topk(logits, self.topk)[1][0]:
                    net.zero_grad()
                    self.optimizer.zero_grad()
                    #label = probs.detach().argmax(-1)
                    #label = k.unsqueeze(0)
                    label = torch.Tensor([k]).cuda().type(torch.int64)
                    #import ipdb; ipdb.set_trace()
                    #label = torch.multinomial(probs[0], 1)

                    if self.right_output == 'loss':
                        loss = F.cross_entropy(logits, label)
                        loss.backward(retain_graph=True)
                    else:
                        raise NotImplementedError

                    self.optimizer.step()
                    nat_grads = [param.grad.data for param in net.parameters()]
                    net.zero_grad()
                    self.optimizer.zero_grad()
                    if self.left_output == 'logits':
                        logits[0][label].backward(retain_graph=True)
                    elif self.left_output == 'loss':
                        loss.backward(retain_graph=True)
                    elif self.left_output == 'softmax':
                        probs[0][label].backward(retain_graph=True)
                    elif self.left_output == 'regret':
                        regret = logits[0][label] + torch.logsumexp(logits[0], dim=0)
                        regret.backward(retain_graph=True)

                    left_grads = [param.grad.data for param in net.parameters()]
                    nak.append(np.sum([(torch.dot(rg.flatten(), ng.flatten()) / ng.numel()).cpu().numpy() for rg, ng in zip(left_grads, nat_grads)]))

            for _ in range(self.num_classes - len(nak)):
                nak.append(0)
            nak = np.array(nak)
            import ipdb; ipdb.set_trace()
        """


class EKFAC(Optimizer):

    def __init__(self, net, eps, niters, sua=False, layer_eps=True):
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

    def step(self, update_stats=True, update_params=True, grads=None, left_grads=None):
        """Performs one step of preconditioning."""
        nak_cum = None

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
                nat_bgrad, nat_wgrad = self._precond_sua_ra(weight, bias, group, state, wgrad, bgrad)
            elif group['layer_type'] == 'BatchNorm2d':
                nat_bgrad, nat_wgrad = self._precond_diag_ra(weight, bias, group, state, wgrad, bgrad)
            else:
                nat_bgrad, nat_wgrad = self._precond_ra(weight, bias, group, state, wgrad, bgrad)

            if nat_wgrad is not None:
                ncls = nat_wgrad.shape[0]
                if left_grads is None:
                    nak = torch.mm(wgrad.reshape(wgrad.shape[0], -1), nat_wgrad.reshape(nat_wgrad.shape[0], -1).t()) / self.numel
                else:
                    nak = torch.mm(left_grads[weight].reshape(left_grads[weight].shape[0], -1), nat_wgrad.reshape(nat_wgrad.shape[0], -1).t()) / self.numel
                if nak_cum is None:
                    nak_cum = nak
                else:
                    nak_cum += nak

                if nat_bgrad is not None:
                    if left_grads is None:
                        nak = torch.mm(bgrad.reshape(bgrad.shape[0], -1), nat_bgrad.reshape(nat_bgrad.shape[0], -1).t()) / self.numel
                    else:
                        nak = torch.mm(left_grads[bias].reshape(left_grads[bias].shape[0], -1), nat_bgrad.reshape(nat_bgrad.shape[0], -1).t()) / self.numel
                    nak_cum += nak
        self._iteration_counter += 1
        return nak_cum

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

    def _precond_ra(self, weight, bias, group, state, g=None, gb=None):
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
        if self.layer_eps:
            eps = self.eps * m2.max()
        else:
            eps = self.eps

        g_nat_kfe = g_kfe / (m2 + eps)
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

    def _precond_sua_ra(self, weight, bias, group, state, g=None, gb=None):
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
        if self.layer_eps:
            eps = self.eps * m2.max()
        else:
            eps = self.eps
        g_nat_kfe = g_kfe / (m2 + eps)
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

    def _precond_diag_ra(self, weight, bias, group, state, g=None, gb=None):
        res = []
        if g is None:
            g = weight.grad.data.unsqueeze(0)
        fw = state['exact_fw']
        if self.layer_eps:
            eps = self.eps * fw.max()
        else:
            eps = self.eps
        g_nat = g / (fw + eps)

        if bias is not None:
            if gb is None:
                gb = bias.grad.data.unsqueeze(0)
            fb = state['exact_fb']
            if self.layer_eps:
                eps = self.eps * fb.max()
            else:
                eps = self.eps
            gb_nat = gb / (fb + eps)

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

    def _update_diagonal(self, group, state):
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']

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

    def _compute_kfe(self, group, state):
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
