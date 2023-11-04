import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import numpy as np

class EKFAC(Optimizer):

    def __init__(self, net, eps, num_classes,
                 sua=False, sud=False, layer_eps=True, eps_type='mean',
                 eigenscale=1.0, featskip=1, layerscale=1,
                 center=False, per_class_m1=False,
                 nfeature_reduce=2000,
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
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self.numel = 0
        self.layer_eps = layer_eps
        self.eps_type = eps_type
        self.eigenscale = eigenscale
        self.featskip = featskip
        self.layerscale = layerscale
        self.center = center
        self.num_classes = num_classes
        self.per_class_m1 = per_class_m1
        self.spectrum_order = None
        self.nfeature_reduce = nfeature_reduce

        self._iteration_counter = 1
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d', 'BatchNorm2d']:
            #if mod_class in ['Linear', 'Conv2d']:
                params = [mod.weight]
                self.numel += mod.weight.numel()
                if mod.bias is not None:
                    params.append(mod.bias)
                    self.numel += mod.bias.numel()
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                if mod_class == 'Conv2d':
                    # Adding gathering filter for convolution
                    if not self.sua:
                        d['gathering_filter_sua'] = self._get_gathering_filter(mod, sua=True)
                    if not self.sud:
                        d['gathering_filter_sud'] = self._get_gathering_filter(mod, sua=False)
                self.params.append(d)

        super(EKFAC, self).__init__(self.params, {})

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

        if self.spectrum_order is None:
            full_spectrum = torch.cat(res)
            self.spectrum_order = torch.argsort(full_spectrum)
            self.bin_size = len(self.spectrum_order) // self.nfeature_reduce
            if len(self.spectrum_order) % self.bin_size == 0:
                self.pad_amount == 0
            else:
                self.pad_amount = (self.bin_size + 1) * self.nfeature_reduce - len(self.spectrum_order)
                self.bin_size += 1

        return res

    def init_hooks(self, net):
        self.clear_hooks()
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d', 'BatchNorm2d']:
                #handle = mod.register_forward_pre_hook(self._save_input)
                #self._fwd_handles.append(handle)
                handle = mod.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                handle = mod.register_forward_hook(self._save_input_output)
                self._fwd_handles.append(handle)

    def step(self, update_stats=True, update_params=True, grads=None, inverse=True, l2=False, return_feats=False, labels=None):
        """Performs one step of preconditioning."""
        assert not (labels is None and self.center), "Require labels for per class centering"
        nat_grads = {}

        eigenfeat = []
        res = 0

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
                wgrad = grads[weight]
                if bias is not None:
                    bgrad = grads[bias]

            if group['layer_type'] == 'Conv2d' and self.sua:
                precond_fn = self._precond_sua_ra
            elif group['layer_type'] == 'BatchNorm2d':
                precond_fn = self._precond_diag_ra
            else:
                precond_fn = self._precond_ra

            # only one of grads or g_kfe should be set
            out = precond_fn(weight, bias, group, state, scale,
                    g=wgrad, gb=bgrad, inverse=inverse, l2=l2, labels=labels)

            if group['layer_type'] != 'BatchNorm2d':
                scale *= self.layerscale

            if l2:
                norm, g_kfe, nat_bgrad, nat_wgrad = out
                res += norm
                if return_feats:
                    eigenfeat.append(g_kfe.flatten())
            else:
                nat_bgrad, nat_wgrad = out

            if nat_wgrad is not None:
                nat_grads[weight] = nat_wgrad
                if nat_bgrad is not None:
                    nat_grads[bias] = nat_bgrad
        if l2:
            if return_feats:
                eigenfeat = torch.cat(eigenfeat)
                eigenfeat = eigenfeat[self.spectrum_order]
                # TODO: pad more elegantly to preserve magnitude of feature?
                eigenfeat = torch.cat((eigenfeat, torch.zeros(self.pad_amount).cuda()))
                eigenfeat = eigenfeat.view(-1, self.bin_size).mean(-1)
                return nat_grads, res, eigenfeat
            else:
                return nat_grads, res
        else:
            return nat_grads

    def dict_dot(self, ldict, rdict, params=None, return_layers=False):
        res = []
        if params is None:
            for i, p in enumerate(rdict.keys()):
                res.append(torch.dot(rdict[p].reshape(-1), ldict[p].reshape(-1)) / self.numel)
        else:
            for i, (k, v) in enumerate(params):
                if v in rdict:
                    res.append(torch.dot(ldict[k].reshape(-1), rdict[v].reshape(-1)) / self.numel)

        if return_layers:
            return res
        else:
            return torch.sum(torch.tensor(res))

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
            g=None, gb=None, inverse=True, l2=False, labels=None):
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
        if self.per_class_m1:
            means = torch.take_along_dim(state['m1'], labels.view(*([-1] + [1] * (g_kfe.ndim - 1))), 0)
            g_kfe -= means

        m2, eps = self.get_m2_and_eps(m2, self.eps)
        m2 *= scale
        # layerwise multiplicative damping
        if inverse:
            if l2:
                res.append((g_kfe**2 * self.eigenscale / (m2 + eps)).sum() / self.numel)
                res.append(g_kfe**2)
            g_nat_kfe = g_kfe * self.eigenscale / (m2 + eps)
        else:
            if l2:
                res.append((g_kfe**2 * (m2 + eps) * self.eigenscale).sum() / self.numel)
                res.append(g_kfe**2)
            g_nat_kfe = g_kfe * ((m2 + eps) * self.eigenscale)

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

    def _precond_sua_ra(self, weight, bias, group, state, scale,
            g=None, gb=None, inverse=True, l2=False, labels=None):
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
        if self.per_class_m1:
            means = torch.take_along_dim(state['m1'], labels.view(*([-1] + [1] * (g_kfe.ndim - 1))), 0)
            g_kfe -= means
        m2, eps = self.get_m2_and_eps(m2, self.eps)
        m2 *= scale

        if inverse:
            if l2:
                res.append((g_kfe**2 * self.eigenscale / (m2 + eps)).sum() / self.numel)
                res.append(g_kfe**2)
            g_nat_kfe = g_kfe * self.eigenscale / (m2 + eps)
        else:
            if l2:
                res.append((g_kfe**2 * (m2 + eps) * self.eigenscale).sum() / self.numel)
                res.append(g_kfe**2)
            g_nat_kfe = g_kfe * ((m2 + eps) * self.eigenscale)

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

        return res

    def update_cache(self, op, **kwargs):
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

            elif op == 'average_moments':
                assert 'weight' in kwargs, "Must provied weight"
                weight = kwargs['weight']
                if group['layer_type'] != 'BatchNorm2d':
                    state['m2'] /= weight

            elif op == 'average_means':
                assert 'weight' in kwargs, "Must provied weight"
                weight = kwargs['weight']
                if group['layer_type'] == 'BatchNorm2d':
                    state['mean_fw'] /= weight
                    state['mean_fb'] /= weight
                else:
                    if group['layer_type'] == 'Conv2d':
                        weight = weight[:, None, None]
                    else:
                        weight = weight[:, None]
                    state['xu'] /= weight
                    state['gyu'] /= weight

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

            elif op == 'moments':
                ex_weight = kwargs.get('ex_weight', None)
                grads = kwargs.get('grads', None)
                labels = kwargs.get('labels', None)
                g = gb = None
                if grads is not None:
                    g = grads[weight]
                    if bias is not None:
                        gb = grads[bias]
                if group['layer_type'] != 'BatchNorm2d':
                    self._update_block_moments(group, state, g=g, gb=gb, ex_weight=ex_weight, labels=labels)

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

    def print_eigens(self):
        print(f"Total number of KFAC params: {self.numel}")
        self.update_cache('print')

    # 0. accumulate mean activation/output gradients
    def update_mean(self, ex_weight=None, labels=None):
        assert not (self.center and labels is None), "Need labels to calculate per class averages"
        self.update_cache('mean', ex_weight=ex_weight, labels=labels)

    def average_means(self, weight):
        self.update_cache('average_means', weight=weight)

    # 1. accumulate average activation/output gradient covariance across batches
    def update_state(self, ex_weight=None, labels=None):
        assert not (self.center and labels is None), "Need labels to calculate per class BN moments"
        self.update_cache('state', ex_weight=ex_weight, labels=labels)

    # 2. average state across batches
    def average_state(self, weight):
        self.update_cache('average_state', weight=weight)

    # 3. compute eigenbasis from state
    def compute_basis(self):
        self.update_cache('basis')

    # 4. reaccumulate second moments
    def update_moments(self, grads=None, ex_weight=None, labels=None):
        assert not (self.center and labels is None), "Need labels to calculate per class eigen moments"
        self.update_cache('moments', grads=grads, ex_weight=ex_weight, labels=labels)

    # 5. average second moments
    def average_moments(self, weight):
        self.update_cache('average_moments', weight=weight)

    def remove_top_moments(self, nremove):
        self.update_cache('remove', nremove=nremove)

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
        if ex_weight is None:
            ex_weight = torch.ones(len(x)).cuda()

        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                # expand channel dimension by filter size
                x = F.conv2d(x, group['gathering_filter_sua'],
                             stride=mod.stride, padding=mod.padding,
                             groups=mod.in_channels) # B x DKK x HW
            x = x.reshape(x.shape[0], x.shape[1], -1) # B x D x HW, accumulate along BHW
        else:
            x = x.data # B x D

        # cat bias 1 along D
        if mod.bias is not None:
            ones = torch.ones_like(x[:, :1])
            x = torch.cat([x, ones], dim=1)

        # accumulate activation mean
        if mean:
            if group['layer_type'] == 'Conv2d':
                x = x * ex_weight[:, None, None]
            else:
                x = x * ex_weight[:, None]
            indices = labels.view(*([-1] + [1] * (x.ndim - 1))).expand_as(x)
            if 'xu' not in state:
                state['xu'] = torch.zeros((self.num_classes, *x.shape[1:])).cuda()
            state['xu'].scatter_add_(0, indices, x) # acumulate D x HW

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

            xxt = torch.mm(xw, x.t()) / float(x.shape[1])
            if 'xxt' not in state:
                state['xxt'] = xxt
            else:
                state['xxt'] += xxt

        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            if not self.sud:
                # expand channel dimension by filter size
                gy = F.conv2d(gy, group['gathering_filter_sud'],
                             stride=mod.stride, padding=mod.padding,
                             groups=mod.out_channels)
            gy = gy.reshape(gy.shape[0], gy.shape[1], -1) # B x D x HW
        else:
            gy = gy.data

        # accumulate output gradient mean
        if mean:
            if group['layer_type'] == 'Conv2d':
                gy = gy * ex_weight[:, None, None]
            else:
                gy = gy * ex_weight[:, None]
            indices = labels.view(*([-1] + [1] * (gy.ndim - 1))).expand_as(gy)
            if 'gyu' not in state:
                state['gyu'] = torch.zeros((self.num_classes, *gy.shape[1:])).cuda() # K x D x HW for conv
            state['gyu'].scatter_add_(0, indices, gy)

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
            ggt = torch.mm(gyw, gy.t()) / float(gy.shape[1])
            if 'ggt' not in state:
                state['ggt'] = ggt
            else:
                state['ggt'] += ggt

    def _update_block_moments(self, group, state, g=None, gb=None, ex_weight=None, labels=None):
        """refit second moments"""
        # g and gb will have an extra batch dimension!!
        isconv = group['layer_type'] == 'Conv2d'
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        if g is None:
            g = group['params'][0].grad.data.unsqueeze(0)
        s = g.shape
        mod = group['mod']
        if isconv and not self.sua:
            g = g.contiguous().view(s[0], s[1], s[2]*s[3]*s[4])
        if len(group['params']) == 2:
            if gb is None:
                gb = group['params'][1].grad.data.unsqueeze(0)
            if isconv and self.sua:
                gb = gb.view(s[0], -1, 1, 1, 1).expand(-1, -1, -1, s[3], s[4])
                g = torch.cat([g, gb], dim=2)
            else:
                g = torch.cat([g, gb.view(gb.shape[0], gb.shape[1], 1)], dim=2)
        if isconv and self.sua:
            g_kfe = self._to_kfe_sua(g, kfe_x, kfe_gy)
        else:
            g_kfe = torch.matmul(torch.matmul(kfe_gy.t(), g), kfe_x)

        if ex_weight is None:
            ex_weight = torch.ones(g_kfe.shape[0]).cuda()
        ex_weight = ex_weight.view(-1, *(1,) * (g_kfe.ndim - 1))

        if 'm2' not in state:
            state['m2'] = (g_kfe.detach()**2 * ex_weight).sum(0)
        else:
            state['m2'] += (g_kfe.detach()**2 * ex_weight).sum(0)

        """
        if self.per_class_m1:
            m1_kfe = g_kfe.detach() * ex_weight
            indices = labels.view(*([-1] + [1] * (m1_kfe.ndim - 1))).expand_as(m1_kfe)
            if 'm1' not in state:
                state['m1'] = torch.zeros((self.num_classes, *m1_kfe.shape[1:])).cuda()
            state['m1'].scatter_add_(0, indices, m1_kfe)
        else:
            if 'm1' not in state:
                state['m1'] = (g_kfe.detach() * ex_weight).sum(0)
            else:
                state['m1'] += (g_kfe.detach() * ex_weight).sum(0)
        """

    def _compute_block_basis(self, group, state, skip_m2=True):
        """computes eigendecomposition."""
        xxt = state['xxt']
        _, state['kfe_x'] = torch.linalg.eigh(xxt.cpu(), UPLO='U')
        state['kfe_x'] = state['kfe_x'].cuda()

        ggt = state['ggt']
        _, state['kfe_gy'] = torch.linalg.eigh(ggt, UPLO='U')
        state['kfe_gy'] = state['kfe_gy'].cuda()

        """
        if not skip_m2:
            state['m2'] = Eg.unsqueeze(1) * Ex.unsqueeze(0) * state['num_locations']
            #print(mod)
            #print(f"Max Eigenvalue: {state['m2'].max()}, Min Eigenvalue: {state['m2'].min()}, Mean Eigenvalue: {state['m2'].mean()}")
            if group['layer_type'] == 'Conv2d' and self.sua:
                ws = group['params'][0].grad.data.size()
                state['m2'] = state['m2'].view(Eg.size(0), Ex.size(0), 1, 1).expand(-1, -1, ws[2], ws[3])
        """

    def _get_gathering_filter(self, mod, sua=True):
        """Convolution filter that extracts input patches."""
        kw, kh = mod.kernel_size
        if sua:
            g_filter = mod.weight.data.new(kw * kh * mod.in_channels, 1, kw, kh)
        else:
            g_filter = mod.weight.data.new(kw * kh * mod.out_channels, 1, kw, kh)
        g_filter.fill_(0)
        for i in range(mod.in_channels):
            for j in range(kw):
                for k in range(kh):
                    g_filter[k + kh*j + kw*kh*i, 0, j, k] = 1
        return g_filter

    def _to_kfe_sua(self, g, vx, vg):
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
