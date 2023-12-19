import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import numpy as np

from typing import Optional, Tuple

class EKFAC(Optimizer):

    def __init__(self, net, eps, num_classes,
                 sua=False, sud=False, layer_eps=True, eps_type='mean',
                 eigenscale=1.0, featskip=1, layerscale=1,
                 center=False, per_class_m1=False,
                 nfeature_reduce=2000, nfeature_layer=50,
    ):
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
        self.sud = sud
        assert (self.sua or self.sud), "Must assume one of SUA or SUD"
        self.layer_eps = layer_eps
        self.eps_type = eps_type
        self.eigenscale = eigenscale
        self.featskip = featskip
        self.layerscale = layerscale
        self.center = center
        self.num_classes = num_classes
        self.per_class_m1 = per_class_m1
        self.spectrum_order = None
        self.layer_order = {}
        self.nfeature_reduce = nfeature_reduce
        self.nfeature_layer = nfeature_layer

        self._iteration_counter = 1
        self.rebuild(net)

        self.prev_hess = None
        self.prev_weight = None
        self.layer_act = None

        self.shortcut_hess = None
        self.shortcut_weight = None
        self.shortcut_layer = None

        self.transfer = []

    def rebuild(self, net):
        self.numel = 0
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        for mod in net.modules():
            mod_class = mod.__class__.__name__

            if issubclass(mod.__class__, nn.Conv2d):
                mod_class = 'Conv2d'

            #if mod_class in ['Linear', 'Conv2d', 'BatchNorm2d']:
            if mod_class in ['Linear', 'Conv2d']:
                params = [mod.weight]
                self.numel += mod.weight.numel()
                if mod.bias is not None:
                    params.append(mod.bias)
                    self.numel += mod.bias.numel()
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                if mod_class == 'Conv2d':
                    if not self.sua:
                        d['gathering_filter_sua'] = self._get_gathering_filter(mod)
                    if not self.sud:
                        d['gathering_filter_sud'] = self._get_gathering_filter(mod, channels=mod.out_channels, reverse=True)
                    #d['x_autocorr_filter'] = self._get_autocorr_filter(mod, mod.in_channels)
                    #d['gy_autocorr_filter'] = self._get_autocorr_filter(mod, mod.out_channels)
                self.params.append(d)

        super(EKFAC, self).__init__(self.params, {})

    def get_thresh_and_idx(self, eigens, nfeatures):
        deigens = eigens + self.eps
        threshes = torch.logspace(torch.log(deigens.min()), torch.log(deigens.max()), nfeatures, math.e)
        scatter_idx = torch.zeros_like(deigens).long()
        pcum = len(scatter_idx)
        scatter_idx = torch.zeros_like(deigens).long()
        for t in threshes:
            tcum = (deigens > t).sum()
            if tcum == pcum:
                continue
            pcum = tcum
            scatter_idx[deigens > t] += 1

        return threshes, scatter_idx


    def get_spectrum(self):
        res = []
        for group in self.param_groups:
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            if 'm2' in state:
                m2 = state['m2'].flatten()
            else:
                m2 = torch.cat((state['exact_fw'].flatten(), state['exact_fb'].flatten()))
            res.append(m2)

            lthresh, lscatter_idx = self.get_thresh_and_idx(m2, self.nfeature_layer)
            self.layer_order[weight] = [lthresh, lscatter_idx.max() + 1, lscatter_idx]

        if self.spectrum_order is None:
            full_spectrum = torch.cat(res)
            self.thresh, self.scatter_idx = self.get_thresh_and_idx(full_spectrum, self.nfeature_reduce)
            print(f"Reducing features from {self.nfeature_reduce} to {self.scatter_idx.max()}")
            self.nfeature_reduce = self.scatter_idx.max()

        return res

    def init_hooks(self, net):
        self.clear_hooks()
        for name, mod in net.named_modules():
            mod_class = mod.__class__.__name__
            if issubclass(mod.__class__, nn.Conv2d):
                mod_class = 'Conv2d'
            if mod_class in ['Linear', 'Conv2d']:
                #handle = mod.register_forward_pre_hook(self._save_input)
                #self._fwd_handles.append(handle)
                #handle = mod.register_full_backward_hook(self._save_grad_output_and_hess(name))
                handle = mod.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                handle = mod.register_forward_hook(self._save_input_output)
                self._fwd_handles.append(handle)
            """
            elif mod_class in ['ReLU', 'AdaptiveAvgPool2d', 'MaxPool2d']:
                handle = mod.register_full_backward_hook(self._track_transfer)
                self._bwd_handles.append(handle)
            """

    def _track_transfer(self, mod, grad_input, grad_output):
        self.transfer.append(mod)

    def step(self, update_stats=True, update_params=True, grads=None, inverse=True, return_feats=False, return_grads=False, labels=None, layers=False, sandwich=False):
        """Performs one step of preconditioning."""
        assert not (labels is None and self.center), "Require labels for per class centering"
        nat_grads = {}

        eigenfeat = []
        #layerfeat = []
        if layers:
            l2_norm = []
            m1_norm = []
            l1_norm = []
        else:
            m2_norm = 0
            m1_norm = 0
            l1_norm = 0

        scale = 1.

        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            wgrad = bgrad = None
            if grads is not None:
                wgrad, wname = grads[weight]
                if bias is not None:
                    bgrad, bname = grads[bias]

            if group['layer_type'] == 'Conv2d' and self.sua and self.sud:
                precond_fn = self._precond_sua_sud_ra
            elif group['layer_type'] == 'BatchNorm2d':
                raise NotImplementedError
                #precond_fn = self._precond_diag_ra
            else:
                precond_fn = self._precond_ra

            # only one of grads or g_kfe should be set
            out = precond_fn(weight, bias, group, state, scale,
                    g=wgrad, gb=bgrad, inverse=inverse, labels=labels, sandwich=sandwich)

            if group['layer_type'] != 'BatchNorm2d':
                scale *= self.layerscale

            _l2_norm, _l1_norm, nat_bgrad, nat_wgrad = out
            if layers:
                l2_norm.append(_l2_norm)
                #m1_norm.append(_m1_norm)
                l1_norm.append(_l1_norm)
            else:
                l2_norm += _l2_norm
                l1_norm += _l1_norm
                #m1_norm += _m1_norm
            if return_feats:
                #_, nfeatures, scatter_idx = self.layer_order[weight]
                #_layerfeat = torch.scatter_add(torch.zeros(nfeatures).cuda(), 0, scatter_idx, g_kfe.flatten())
                eigenfeat.append(g_kfe.flatten())
                #layerfeat.append(_layerfeat)

            if return_grads:
                if nat_wgrad is not None:
                    nat_grads[wname] = nat_wgrad
                    if nat_bgrad is not None:
                        nat_grads[bname] = nat_bgrad

        #res = [l2_norm, l1_norm, m1_norm]
        res = [l2_norm, l1_norm]

        if return_feats:
            eigenfeat = torch.cat(eigenfeat)
            out = torch.zeros(self.nfeature_reduce + 1).cuda()
            eigenfeat = torch.scatter_add(out, 0, self.scatter_idx, eigenfeat)
            res.append(eigenfeat)
        if return_grads:
            res.append(nat_grads)

        return res

    def dict_dot(self, ldict, rdict, params=None, return_layers=False):
        res = []
        sres = 0
        if params is None:
            for i, p in enumerate(rdict.keys()):
                if return_layers:
                    res.append(torch.dot(rdict[p].reshape(-1), ldict[p].reshape(-1)) / self.numel)
                else:
                    sres += torch.dot(rdict[p].reshape(-1), ldict[p].reshape(-1)) / self.numel
        else:
            for i, (k, v) in enumerate(params):
                if v in rdict:
                    if return_layers:
                        res.append(torch.dot(ldict[k].reshape(-1), rdict[v].reshape(-1)) / self.numel)
                    else:
                        sres += torch.dot(ldict[k].reshape(-1), rdict[v].reshape(-1)) / self.numel

        if return_layers:
            return res
        else:
            return sres

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
        self.state[mod]['x'] = i[0].detach()

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]['gy'] = grad_output[0].detach()

    def _save_grad_output_and_hess(self, name):
        def hook_fn(mod, grad_input, grad_output):
            """Saves grad on output of layer to compute covariance."""
            self.state[mod]['gy'] = grad_output[0].detach()

            if isinstance(mod, nn.BatchNorm2d):
                raise NotImplementedError

            if 'shortcut' in name:
                self.shortcut_layer = int(name.split('layer')[1].split('.')[0]) - 1

            if self.prev_weight is None:
                # top layer, just copy the output hessian
                if 'hy' not in self.state[mod]:
                    self.state[mod]['hy'] = self.prev_hess.sum(0)
                else:
                    self.state[mod]['hy'] += self.prev_hess.sum(0)
            else:
                # recurse the hessian calculation
                j_act = self.state[mod]['y']

                # manually backprop through transfer functions
                for t_mod in self.transfer[::-1]:
                    t_name = t_mod.__class__.__name__
                    if t_name == 'AdaptiveAvgPool2d':
                        j_act = j_act.mean(-1).mean(-1)

                    elif t_name == 'ReLU':
                        j_act = (j_act > 0).float()

                    elif t_name == 'MaxPool2d':
                        # divide by kernel size since we average over locations anyways
                        j_act = t_mod(j_act) / t_mod.kernel_size**2

                # TODO: reshape activations to filter patches?
                """
                if j_act.ndim == 4:
                    if j_act.shape[-1] == 1 and j_act.shape[-2] == 1:
                        j_act = j_act[..., 0, 0]
                    else:
                        j_act = F.conv2d(j_act, self.state[mod]['gathering_filter_hy'],
                                         stride=self.prev_mod.stride, padding=self.prev_mod.padding,
                                         groups=self.prev_mod.in_channels) # B x DKK x HW
                        j_act = j_act.mean(-1).mean(-1)

                # TODO: if conv reshape weight to IKK x O, average across filter?
                if self.prev_weight.ndim == 4:
                    O, I, K, _ = self.prev_weight.shape
                    self.prev_weight = self.prev_weight.view(self.prev_weight.shape[0], -1)
                    hfactor = torch.matmul(torch.diag_embed(j_act), self.prev_weight.transpose(-1, -2))
                    hfactor = hfactor.view(hfactor.shape[0], I, K, K, hfactor.shape[-1]).mean(2).mean(2)
                else:
                    hfactor = torch.matmul(torch.diag_embed(j_act), self.prev_weight.transpose(-1, -2))
                """
                if j_act.ndim == 4:
                    j_act = j_act.mean(-1).mean(-1)

                if self.prev_weight.ndim == 4:
                    self.prev_weight = self.prev_weight.mean(-1).mean(-1)

                hfactor = torch.matmul(torch.diag_embed(j_act), self.prev_weight.transpose(-1, -2))

                hess = torch.bmm(torch.bmm(hfactor, self.prev_hess), hfactor.transpose(-1, -2))

                # hack to keep track of hessians from the shortcut connection
                if self.shortcut_layer is not None and name == 'layer' + str(self.shortcut_layer) + '.1.conv2':
                    # assume only relu on shortcut connection

                    if self.shortcut_weight.ndim == 4:
                        self.shortcut_weight = self.shortcut_weight.mean(-1).mean(-1)

                    s_hfactor = torch.matmul(torch.diag_embed(j_act), self.shortcut_weight.transpose(-1, -2))
                    s_hess = torch.bmm(torch.bmm(s_hfactor, self.shortcut_hess), s_hfactor. transpose(-1, -2))
                    hess += s_hess
                    self.shortcut_hess = None
                    self.shortcut_weight = None
                    self.shortcut_layer = None

                if 'hy' not in self.state[mod]:
                    self.state[mod]['hy'] = hess.sum(0)
                else:
                    self.state[mod]['hy'] += hess.sum(0)

                if 'shortcut' in name:
                    self.shortcut_hess = hess
                else:
                    self.prev_hess = hess


            prev_weight = mod.weight.detach()
            self.prev_act = self.state[mod]['x']
            if 'shortcut' in name:
                self.shortcut_weight = prev_weight
            else:
                self.prev_weight = prev_weight

            self.transfer = []
            self.prev_mod = mod

        return hook_fn


    def _save_input_output(self, mod, i, o):
        self.state[mod]['x'] = i[0].detach()
        self.state[mod]['y'] = o.detach()

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

    def get_m2_and_eps(self, m2, eps):
        if self.eps_type == 'replace':
            m2[m2 < self.eps] = self.eps
            eps = 0
        elif self.eps_type == 'abs_replace':
            m2 = torch.abs(m2)
            m2[m2 == 0] = self.eps
            eps = 0
        else:
            eps = self.get_eps(m2)
        return m2, eps

    def _precond_ra(self, weight, bias, group, state, scale,
            g=None, gb=None, inverse=True, labels=None, sandwich=False):
        """Applies preconditioning."""
        res = []
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        #kfe_hy = state['kfe_hy']
        m2 = state['m2']
        #m1 = state['m1']
        if g is None:
            g = weight.grad.data.unsqueeze(0)
        s = g.shape
        #bs = self.state[group['mod']]['x'].size(0)
        if group['layer_type'] == 'Conv2d':
            if not self.sud:
                g = g.contiguous().permute(0, 1, 3, 4, 2).reshape(s[0], s[1]*s[3]*s[4], s[2]) # B x OHW x I
            elif not self.sua:
                g = g.contiguous().view(s[0], s[1], s[2]*s[3]*s[4]) # B x O x IHW
            else:
                raise NotImplementedError

        if bias is not None:
            if gb is None:
                gb = bias.grad.data.unsqueeze(0) # B x O
            if group['layer_type'] == 'Conv2d' and not self.sud:
                g = torch.cat([g, gb.view(s[0], s[1], 1, 1).expand(1, s[1], s[3], s[4]).reshape(s[0], -1, 1)], dim=2) # B x OHW x (I + 1)
            else:
                g = torch.cat([g, gb.view(gb.shape[0], gb.shape[1], 1)], dim=2) # B X O x (IHW + 1)

        if sandwich:
            hm2, eps = self.get_m2_and_eps(state['hm2'], self.eps)
            g_kfe = torch.matmul(torch.matmul(kfe_hy.t(), g), kfe_x)
            g_kfe = g_kfe / (hm2 + eps)
            g_kfe = torch.matmul(torch.matmul(kfe_hy, g_kfe), kfe_x.t())
            g_kfe = torch.matmul(torch.matmul(kfe_gy.t(), g), kfe_x)
        else:
            g_kfe = torch.matmul(torch.matmul(kfe_gy.t(), g), kfe_x)

        m2, eps = self.get_m2_and_eps(m2, self.eps)
        m2 *= scale

        # layerwise multiplicative damping
        if inverse:
            res.append((g_kfe**2 * self.eigenscale / (m2 + eps)).sum() )
            res.append((torch.abs(g_kfe) * self.eigenscale / torch.sqrt(m2 + eps)).sum())
            #res.append((g_kfe * m1 * self.eigenscale * (m2 + eps)).sum())
            #res.append(g_kfe**2 * self.eigenscale / (m2 + eps))
            g_nat_kfe = g_kfe * self.eigenscale / (m2 + eps)
        else:
            res.append((g_kfe**2 * self.eigenscale * (m2 + eps)).sum() )
            res.append((torch.abs(g_kfe) * self.eigenscale * torch.sqrt(m2 + eps)).sum())
            #res.append((g_kfe * m1 * self.eigenscale * (m2 + eps)).sum() )
            #res.append(g_kfe**2 * self.eigenscale * (m2 + eps) )
            g_nat_kfe = g_kfe * self.eigenscale * (m2 + eps)

        g_nat = torch.matmul(torch.matmul(kfe_gy, g_nat_kfe), kfe_x.t())

        # just skip for sud since we really don't care about the parameter space...
        if bias is not None and self.sud:
            gb = g_nat[:, :, -1].contiguous().view(s[0], *bias.shape)
            if bias.grad is not None:
                bias.grad.data = gb
                res.append(None)
            else:
                res.append(gb)
            g_nat = g_nat[:, :, :-1]
        else:
            res.append(None)

        if not self.sud:
            res.append(None)
        else:
            g_nat = g_nat.contiguous().view(*s)
            if weight.grad is not None:
                weight.grad.data = g_nat
                res.append(None)
            else:
                res.append(g_nat)

        return res

    def _precond_sua_sud_ra(self, weight, bias, group, state, scale,
            g=None, gb=None, inverse=True, labels=None, sandwich=False):
        """Preconditioning for KFAC SUA."""
        res = []
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        #kfe_hy = state['kfe_hy']
        m2 = state['m2']
        #m1 = state['m1']
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

        if sandwich:
            hm2, eps = self.get_m2_and_eps(state['hm2'], self.eps)
            g_kfe = self._to_kfe_sua(g, kfe_x, kfe_hy)
            g_kfe = g_kfe / (hm2 + eps)
            g_kfe = self._to_kfe_sua(g_kfe, kfe_x.t(), kfe_hy.t())
            g_kfe = self._to_kfe_sua(g_kfe, kfe_x, kfe_gy)
        else:
            g_kfe = self._to_kfe_sua(g, kfe_x, kfe_gy)

        m2, eps = self.get_m2_and_eps(m2, self.eps)
        m2 *= scale

        if inverse:
            res.append((g_kfe**2 * self.eigenscale / (m2 + eps)).sum() )
            res.append((torch.abs(g_kfe) * self.eigenscale / torch.sqrt(m2 + eps)).sum())
            #res.append((g_kfe * m1 * self.eigenscale * (m2 + eps)).sum() )
            #res.append(g_kfe**2 * self.eigenscale / (m2 + eps) )
            g_nat_kfe = g_kfe * self.eigenscale / (m2 + eps)
        else:
            res.append((g_kfe**2 * self.eigenscale * (m2 + eps)).sum() )
            res.append((torch.abs(g_kfe) * self.eigenscale * torch.sqrt(m2 + eps)).sum())
            #res.append((g_kfe * m1 * self.eigenscale * (m2 + eps)).sum() )
            #res.append(g_kfe**2 * self.eigenscale * (m2 + eps) )
            g_nat_kfe = g_kfe * self.eigenscale * (m2 + eps)

        g_nat = self._to_kfe_sua(g_nat_kfe, kfe_x.t(), kfe_gy.t())
        if bias is not None:
            gb = g_nat[:, :, -1, s[3]//2, s[4]//2]
            if bias.grad is not None:
                #bias.grad.data = gb.squeeze()
                res.append(None)
            else:
                res.append(gb)
            g_nat = g_nat[:, :, :-1]
        else:
            res.append(None)

        if weight.grad is not None:
            #weight.grad.data = g_nat.squeeze()
            res.append(None)
        else:
            res.append(g_nat)

        return res

    """
    def _precond_diag_ra(self, weight, bias, group, state, scale,
            g=None, gb=None, inverse=True, l2=False, labels=None):
        res = []
        if g is None:
            g = weight.grad.data.unsqueeze(0)

        if self.per_class_m1:
            means = torch.take_along_dim(state['mean_fw'], labels.view(*([-1] + [1] * (g.ndim - 1))), 0)
            g -= means
        fw = state['exact_fw']
        fw, eps = self.get_m2_and_eps(fw, self.eps)
        fw *= scale

        if inverse:
            if l2:
                g_nat_raw = g**2
            g_nat = g * self.eigenscale / (fw + eps)
        else:
            if l2:
                g_nat_raw = g**2
            g_nat = g * ((fw + eps) * self.eigenscale)

        if bias is not None:
            if gb is None:
                gb = bias.grad.data.unsqueeze(0)
            if self.per_class_m1:
                means = torch.take_along_dim(state['mean_fb'], labels.view(*([-1] + [1] * (gb.ndim - 1))), 0)
                gb -= means
            fb = state['exact_fb']
            fb, eps = self.get_m2_and_eps(fb, self.eps)
            fb *= scale

            if inverse:
                if l2:
                    gb_nat_raw = gb**2
                gb_nat = gb * self.eigenscale / (fb + eps)
            else:
                if l2:
                    gb_nat_raw = gb**2
                gb_nat = gb * ((fb + eps) * self.eigenscale)

            if bias.grad is not None:
                bias.grad.data = gb_nat.squeeze()
                res.append(None)
            else:
                res.append(gb_nat)
        else:
            res.append(None)

        if l2:
            res.append((g_nat.sum() + gb_nat.sum()) / self.numel)
            res.append(torch.cat((g_nat_raw, gb_nat_raw), -1))

        if weight.grad is not None:
            weight.grad.data = g_nat.squeeze()
            res.append(None)
        else:
            res.append(g_nat)

        if l2:
            res = [res[1], res[2], res[0], res[3]]

        return res
    """

    def update_cache(self, op, **kwargs):
        # return things if we need to
        res = []

        for group in self.param_groups:
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]

            if op == 'average_state':
                assert 'weight' in kwargs, "Must provied weight"
                weight = kwargs['weight']
                if group['layer_type'] == 'BatchNorm2d':
                    state['exact_fw'] /= weight
                    state['exact_fb'] /= weight
                else:
                    state['xxt'] /= weight
                    state['ggt'] /= weight
                    #state['hy'] = self.state[group['mod']]['hy'] / weight

            elif op == 'average_moments':
                assert 'weight' in kwargs, "Must provied weight"
                weight = kwargs['weight']
                moment = kwargs.get('moment', None)
                if group['layer_type'] != 'BatchNorm2d':
                    if moment == 2:
                        state['m2'] /= weight
                    elif moment == 1:
                        state['m1'] /= weight
                    else:
                        raise NotImplementedError

            elif op == 'state':
                ex_weight = kwargs.get('ex_weight', None)
                labels = kwargs.get('labels', None)
                if group['layer_type'] == 'BatchNorm2d':
                    self._update_diagonal(group, state, ex_weight=ex_weight, labels=labels)
                else:
                    self._update_block(group, state, ex_weight=ex_weight, labels=labels)

            elif op == 'mean':
                ex_weight = kwargs.get('ex_weight', None)
                labels = kwargs.get('labels', None)
                if group['layer_type'] == 'BatchNorm2d':
                    self._update_diagonal(group, state, ex_weight=ex_weight, labels=labels, mean=True)
                else:
                    self._update_block(group, state, ex_weight=ex_weight, labels=labels, mean=True)

            elif 'moments' in op:
                ex_weight = kwargs.get('ex_weight', None)
                grads = kwargs.get('grads', None)
                labels = kwargs.get('labels', None)
                moment = kwargs.get('moment', None)
                g = gb = None
                if grads is not None:
                    g = grads[weight]
                    if bias is not None:
                        gb = grads[bias]
                if group['layer_type'] != 'BatchNorm2d':
                    self._update_block_moments(group, state, moment, g=g, gb=gb, ex_weight=ex_weight, labels=labels)

            elif op == 'basis':
                if group['layer_type'] != "BatchNorm2d":
                    self._compute_block_basis(group, state)

            elif op == 'remove':
                if group['layer_type'] != "BatchNorm2d":
                    eigens = state['m2'].view(-1)
                    ridx = torch.topk(eigens, kwargs['nremove'])[1]
                    eigens[ridx] = torch.inf
                    print(f"Removing {kwargs['nremove']} eigenvalues of {len(eigens)}")

            elif op == 'print':
                print(group['mod'])
                if group['layer_type'] == 'BatchNorm2d':
                    eigens = state['exact_fw']
                    #means = state['mean_fw']
                else:
                    eigens = state['m2']
                    #means = state['m1']

                print(f"2nd Moments | Max: {eigens.max()},\t Min: {eigens.min()},\t Min Nonzero: {eigens[eigens != 0.0].min()},\t Mean: {eigens.mean()}")
                #print(f"1st Moments | Max: {means.max()},\t Min: {means.min()},\t Min Nonzero: {means[means != 0.0].min()},\t Mean: {means.mean()}")

            else:
                raise Exception(f"Operation {op} not supported")

        return res

    def print_eigens(self):
        print(f"Total number of KFAC params: {self.numel}")
        self.update_cache('print')

    # 0. accumulate mean activation/output gradients
    def update_mean(self, ex_weight=None, labels=None):
        assert not (self.center and labels is None), "Need labels to calculate per class averages"
        return self.update_cache('mean', ex_weight=ex_weight, labels=labels)

    # 1. accumulate average activation/output gradient covariance across batches
    def update_state(self, ex_weight=None, labels=None):
        assert not (self.center and labels is None), "Need labels to calculate per class BN moments"
        return self.update_cache('state', ex_weight=ex_weight, labels=labels)

    # 2. average state across batches
    def average_state(self, weight):
        return self.update_cache('average_state', weight=weight)

    # 3. compute eigenbasis from state
    def compute_basis(self):
        return self.update_cache('basis')

    # 4. reaccumulate second moments
    def update_second_moments(self, grads=None, ex_weight=None, labels=None):
        assert not (self.center and labels is None), "Need labels to calculate per class eigen moments"
        return self.update_cache('moments', grads=grads, ex_weight=ex_weight, labels=labels, moment=2)

    # 4a. reaccumulate first moments for validation data
    def update_first_moments(self, grads=None, ex_weight=None, labels=None):
        assert not (self.center and labels is None), "Need labels to calculate per class eigen moments"
        return self.update_cache('moments', grads=grads, ex_weight=ex_weight, labels=labels, moment=1)

    # 5. average second moments
    def average_second_moments(self, weight):
        return self.update_cache('average_moments', weight=weight, moment=2)

    # 5a. average first moments
    def average_first_moments(self, weight):
        return self.update_cache('average_moments', weight=weight, moment=1)

    def remove_top_moments(self, nremove):
        return self.update_cache('remove', nremove=nremove)

    def _update_diagonal(self, group, state, ex_weight=None, labels=None, mean=False):
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        if ex_weight is None:
            ex_weight = torch.ones(len(x)).cuda()

        gw = torch.sum(x * gy, (-1, -2))
        if mean:
            gw_mean = (gw * ex_weight[:, None])
            indices = labels.view(*([-1] + [1] * (gw.ndim - 1))).expand_as(gw_mean)
            if 'mean_fw' not in state:
                state['mean_fw'] = torch.zeros((self.num_classes, *gw.shape[1:])).cuda()
            state['mean_fw'].scatter_add_(0, indices, gw_mean)
        else:
            if self.center:
                gw -= state['mean_fw']
            gw = ((gw**2) * ex_weight[:, None]).sum(0)
            if 'exact_fw' not in state:
                state['exact_fw'] = gw
            else:
                state['exact_fw'] += gw

        if mod.bias is not None:
            gb = torch.sum(gy, (-1, -2))

            if mean:
                gb_mean = (gb * ex_weight[:, None])
                indices = labels.view(*([-1] + [1] * (gb.ndim - 1))).expand_as(gb_mean)
                if 'mean_fb' not in state:
                    state['mean_fb'] = torch.zeros((self.num_classes, *gb.shape[1:])).cuda()
                state['mean_fb'].scatter_add_(0, indices, gb_mean)
            else:
                if self.center:
                    gb -= state['mean_fb']
                gb = (gb**2 * ex_weight[:, None]).sum(0)
                if 'exact_fb' not in state:
                    state['exact_fb'] = gb
                else:
                    state['exact_fb'] += gb

    def _update_block(self, group, state, ex_weight=None, mean=False, labels=None):
        """stores covariances"""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']

        if group['layer_type'] == 'Conv2d':
            state['num_locations'] = x.shape[2] * x.shape[3]
        else:
            state['num_locations'] = 1

        if ex_weight is None:
            ex_weight = torch.ones(len(x)).cuda()

        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            nlocations = x.shape[-1] * x.shape[-2]

            # expand channel dimension by filter size
            """
            x_first = x[:, :1] # grab only first feature layer
            x_patches = F.conv2d(x, group['x_autocorr_filter'], padding='same',
                         groups=mod.in_channels) # B x DKK x HW
            x_patches = x_patches[:, :49] # grab first feature patches
            x_first = x_first * ex_weight.view(*([-1] + [1] * (x_first.ndim - 1))).expand_as(x_first)
            x_patches = x_patches * ex_weight.view(*([-1] + [1] * (x_patches.ndim - 1))).expand_as(x_patches)
            x_first = x_first.transpose(0, 1).reshape(x_first.shape[1], -1)
            x_patches = x_patches.transpose(0, 1).reshape(x_patches.shape[1], -1)
            autocorr = torch.mm(x_first, x_patches.t()) / float(x_patches.shape[1])
            if 'x_autocorr' not in state:
                state['x_autocorr'] = autocorr
            else:
                state['x_autocorr'] += autocorr
            """

            if not self.sua:
                x = F.conv2d(x, group['gathering_filter_sua'],
                             stride=mod.stride, padding=mod.padding,
                             groups=mod.in_channels) # B x DKK x HW
            x = x.reshape(x.shape[0], x.shape[1], -1) # B x D x HW, accumulate along BHW
        else:
            x = x.data # B x D
            nlocations = 1

        # cat bias 1 along D
        if mod.bias is not None:
            ones = torch.ones_like(x[:, :1])
            x = torch.cat([x, ones], dim=1)

        # accumulate activation mean
        if mean:
            if group['layer_type'] == 'Conv2d':
                if 'ex' not in state:
                    # B x D * B x 1
                    state['ex'] = (x.mean(-1) * ex_weight[:, None]).sum(0)
                else:
                    state['ex'] += (x.mean(-1) * ex_weight[:, None]).sum(0)

                if 'ex2' not in state:
                    state['ex2'] = ((x**2).mean(-1) * ex_weight[:, None]).sum(0)
                else:
                    state['ex2'] += ((x**2).mean(-1) * ex_weight[:, None]).sum(0)

        # accumulate activation second moments
        else:
            if self.center:
                means = torch.take_along_dim(state['xu'], labels.view(*([-1] + [1] * (x.ndim - 1))), 0)
                x = x - means

            x_weight = ex_weight.view(*([-1] + [1] * (x.ndim - 1))).expand_as(x)
            xw = x * x_weight

            if group['layer_type'] == 'Conv2d':
                x = x.transpose(0, 1).reshape(x.shape[1], -1) # D x BHW for accumulation
                xw = xw.transpose(0, 1).reshape(xw.shape[1], -1)
            else:
                x = x.t() # D x B for linear
                xw = xw.t()

            xxt = torch.mm(xw, x.t()) / nlocations
            if 'xxt' not in state:
                state['xxt'] = xxt
            else:
                state['xxt'] += xxt


        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            # expand channel dimension by filter size
            """
            gy_first = gy[:, :1] # grab only first feature layer
            gy_patches = F.conv2d(gy, group['gy_autocorr_filter'], padding='same',
                         groups=mod.out_channels) # B gy DKK gy HW
            gy_patches = gy_patches[:, :49] # grab first feature patches
            gy_first = gy_first * ex_weight.view(*([-1] + [1] * (gy_first.ndim - 1))).expand_as(gy_first)
            gy_patches = gy_patches * ex_weight.view(*([-1] + [1] * (gy_patches.ndim - 1))).expand_as(gy_patches)
            gy_first = gy_first.transpose(0, 1).reshape(gy_first.shape[1], -1)
            gy_patches = gy_patches.transpose(0, 1).reshape(gy_patches.shape[1], -1)
            autocorr = torch.mm(gy_first, gy_patches.t()) / float(gy_patches.shape[1])
            if 'gy_autocorr' not in state:
                state['gy_autocorr'] = autocorr
            else:
                state['gy_autocorr'] += autocorr
            """
            nlocations = gy.shape[-1] * gy.shape[-2]
            if not self.sud:
                gy = F.conv2d(gy, group['gathering_filter_sud'],
                              stride=1, padding='same',
                              groups=mod.out_channels) # B x DKK x HW

            gy = gy.reshape(gy.shape[0], gy.shape[1], -1) # B x D x HW
        else:
            nlocations = 1
            gy = gy.data

        # accumulate output gradient mean
        if mean:
            if group['layer_type'] == 'Conv2d':
                if 'egy' not in state:
                    # B x D * B x 1
                    state['egy'] = (gy.mean(-1) * ex_weight[:, None]).sum(0)
                else:
                    state['egy'] += (gy.mean(-1) * ex_weight[:, None]).sum(0)

                if 'egy2' not in state:
                    state['egy2'] = ((gy**2).mean(-1) * ex_weight[:, None]).sum(0)
                else:
                    state['egy2'] += ((gy**2).mean(-1) * ex_weight[:, None]).sum(0)

        # accumulate output gradient second moments
        else:
            if self.center:
                means = torch.take_along_dim(state['gyu'], labels.view(*([-1] + [1] * (gy.ndim - 1))), 0)
                gy = gy - means

            gy_weight = ex_weight.view(*([-1] + [1] * (gy.ndim - 1))).expand_as(gy)
            gyw = gy * gy_weight

            if group['layer_type'] == 'Conv2d':
                gy = gy.transpose(0, 1).reshape(gy.shape[1], -1) # D x BHW for accumulation
                gyw = gyw.transpose(0, 1).reshape(gyw.shape[1], -1)
            else:
                gy = gy.t() # D x B for linear
                gyw = gyw.t()

            # average over locations but sum over batches
            ggt = torch.mm(gyw, gy.t()) / nlocations

            if 'ggt' not in state:
                state['ggt'] = ggt
            else:
                state['ggt'] += ggt

    def _update_block_moments(self, group, state, moment, g=None, gb=None, ex_weight=None, labels=None):
        """refit second moments"""
        # g and gb will have an extra batch dimension!!
        isconv = group['layer_type'] == 'Conv2d'
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        #kfe_hy = state['kfe_hy']
        if g is None:
            g = group['params'][0].grad.data.unsqueeze(0)
        s = g.shape
        mod = group['mod']

        if isconv and not self.sua:
            g = g.contiguous().view(s[0], s[1], s[2]*s[3]*s[4]) # B x O x IHW

        if isconv and not self.sud:
            g = g.contiguous().permute(0, 1, 3, 4, 2).reshape(s[0], s[1]*s[3]*s[4], s[2]) # B x OHW x I

        if len(group['params']) == 2:
            if gb is None:
                gb = group['params'][1].grad.data.unsqueeze(0) # B x O
            if isconv and not self.sua:
                gb = gb.view(s[0], -1, 1)
                g = torch.cat([g, gb], dim=2) # B x O x (IHW + 1)
            elif isconv and not self.sud:
                gb = gb.view(s[0], -1, 1, 1, 1).expand(-1, -1, s[3], s[4], -1).reshape(s[0], s[1]*s[3]*s[4], 1) # B x OHW x 1
                g = torch.cat([g, gb], dim=2) # B x OHW x (I+1)
            elif isconv and self.sua:
                gb = gb.view(s[0], -1, 1, 1, 1).expand(-1, -1, -1, s[3], s[4])
                g = torch.cat([g, gb], dim=2)
            else:
                g = torch.cat([g, gb.view(gb.shape[0], gb.shape[1], 1)], dim=2)

        if isconv and self.sua and self.sud:
            g_kfe = self._to_kfe_sua(g, kfe_x, kfe_gy)
        else:
            g_kfe = torch.matmul(torch.matmul(kfe_gy.t(), g), kfe_x)

        if ex_weight is None:
            ex_weight = torch.ones(g_kfe.shape[0]).cuda()
        ex_weight = ex_weight.view(-1, *(1,) * (g_kfe.ndim - 1))

        if moment == 2:
            if 'm2' not in state:
                state['m2'] = (g_kfe.detach()**2 * ex_weight).sum(0)
            else:
                state['m2'] += (g_kfe.detach()**2 * ex_weight).sum(0)
        elif moment == 1:
            if 'm1' not in state:
                state['m1'] = (g_kfe.detach() * ex_weight).sum(0)
            else:
                state['m1'] += (g_kfe.detach() * ex_weight).sum(0)
        else:
            raise NotImplementedError


    def _compute_block_basis(self, group, state, skip_m2=True):
        """computes eigendecomposition."""
        xxt = state['xxt']
        _, state['kfe_x'] = torch.linalg.eigh(xxt, UPLO='U')
        state['kfe_x'] = state['kfe_x'].cuda()

        ggt = state['ggt']
        _, state['kfe_gy'] = torch.linalg.eigh(ggt, UPLO='U')
        state['kfe_gy'] = state['kfe_gy'].cuda()

    def _get_autocorr_filter(self, mod, channels):
        return self._get_gathering_filter(mod, channels, 7, 7)

    def _get_gathering_filter(self, mod, channels=None, kw=None, kh=None, reverse=False):
        """Convolution filter that extracts input patches."""
        if channels is None:
            channels = mod.in_channels
        if kw is None:
            kw = mod.kernel_size[0]
        if kh is None:
            kh = mod.kernel_size[1]
        g_filter = mod.weight.data.new(kw * kh * channels, 1, kw, kh)
        g_filter.fill_(0)
        for i in range(channels):
            for j in range(kw):
                for k in range(kh):
                    if reverse:
                        g_filter[(kh - 1 - k) + kh*(kw - 1 - j) + kw*kh*i, 0, j, k] = 1
                    else:
                        g_filter[k + kh*j + kw*kh*i, 0, j, k] = 1
        return g_filter

    def _to_kfe_sua(self, g, vx, vg): # B x O x I x H x W, I x I, O x O
        """Project g to the kfe"""
        # NOTE: g includes an extra first dimension for classes
        sg = g.size()
        g = torch.matmul(vg.t(), g.reshape(sg[0], sg[1], -1)).view(sg[0], vg.size(1), sg[2], sg[3], sg[4])
        g = torch.matmul(g.permute(0, 1, 3, 4, 2).contiguous().view(sg[0], -1, sg[2]), vx)
        g = g.view(sg[0], vg.size(1), sg[3], sg[4], vx.size(1)).permute(0, 1, 4, 2, 3)
        return g

    def clear_hooks(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
        for state in self.state.values():
            if 'x' in state:
                del state['x']
                del state['gy']

    def clear_cache(self):
        for state in self.state.values():
            if 'xxt' in state:
                del state['xxt']
                del state['ggt']

    def __del__(self):
        self.clear_hooks()
        self.clear_cache()
