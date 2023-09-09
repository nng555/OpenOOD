import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

class EKFAC(Optimizer):

    def __init__(self, net, eps, sua=False, layer_eps=True, eps_type='mean'):
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

    def init_hooks(self, net):
        self.clear_hooks()
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d', 'BatchNorm2d']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)

    def step(self, update_stats=True, update_params=True, grads=None):
        """Performs one step of preconditioning."""
        nat_grads = {}
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
                nat_bgrad, nat_wgrad = self._precond_sua_ra(weight, bias, group, state, wgrad, bgrad)
            elif group['layer_type'] == 'BatchNorm2d':
                nat_bgrad, nat_wgrad = self._precond_diag_ra(weight, bias, group, state, wgrad, bgrad)
            else:
                nat_bgrad, nat_wgrad = self._precond_ra(weight, bias, group, state, wgrad, bgrad)

            if nat_wgrad is not None:
                nat_grads[weight] = nat_wgrad
                if nat_bgrad is not None:
                    nat_grads[bias] = nat_bgrad

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
        self.state[mod]['x'] = i[0].detach()

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]['gy'] = grad_output[0].detach() * grad_output[0].size(0)

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
        m2, eps = self.get_m2_and_eps(m2, self.eps)
        # layerwise multiplicative damping
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

        m2, eps = self.get_m2_and_eps(m2, self.eps)

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
        fw, eps = self.get_m2_and_eps(fw, self.eps)

        g_nat = g / (fw + eps)

        if bias is not None:
            if gb is None:
                gb = bias.grad.data.unsqueeze(0)
            fb = state['exact_fb']
            fb, eps = self.get_m2_and_eps(fb, self.eps)

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

    def update_cache(self, op, nbatches=1, grads=None):
        for group in self.param_groups:
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            if op == 'average_state':
                if group['layer_type'] == 'BatchNorm2d':
                    state['exact_fw'] /= nbatches
                    state['exact_fb'] /= nbatches
                else:
                    state['xxt'] /= nbatches
                    state['ggt'] /= nbatches
            elif op == 'average_moments':
                if group['layer_type'] != 'BatchNorm2d':
                    state['m2'] /= nbatches
            elif op == 'state':
                if group['layer_type'] == 'BatchNorm2d':
                    self._update_diagonal(group, state)
                else:
                    self._update_block(group, state)
            elif op == 'moments':
                if group['layer_type'] != 'BatchNorm2d':
                    g = gb = None
                    if grads is not None:
                        g = grads[weight]
                        if bias is not None:
                            gb = grads[bias]
                    self._update_block_moments(group, state, g=g, gb=gb)
            elif op == 'basis':
                if group['layer_type'] != "BatchNorm2d":
                    self._compute_block_basis(group, state)
            elif op == 'print':
                print(group['mod'])
                if group['layer_type'] == 'BatchNorm2d':
                    eigens = state['exact_fw']
                else:
                    eigens = state['m2']
                print(f"Max Eigenvalue: {eigens.max()}, Min Eigenvalue: {eigens.min()}, Mean Eigenvalue: {eigens.mean()}")

    def print_eigens(self):
        self.update_cache('print')

    # 1. accumulate average activation/gradients across batches
    def update_state(self):
        self.update_cache('state')

    # 2. average state across batches
    def average_state(self, nbatches):
        self.update_cache('average_state', nbatches)

    # 3. compute eigenbasis from state
    def compute_basis(self):
        self.update_cache('basis')

    # 4. reaccumulate second moments
    def update_moments(self, grads=None):
        self.update_cache('moments', grads=grads)

    # 5. average second moments
    def average_moments(self, nbatches):
        self.update_cache('average_moments', nbatches)

    def _update_diagonal(self, group, state):
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']

        gw = torch.sum(x * gy, (-1, -2))
        gw = (gw**2).mean(0)
        if 'exact_fw' not in state:
            state['exact_fw'] = gw
        else:
            state['exact_fw'] += gw
        if mod.bias is not None:
            gb = torch.sum(gy, (-1, -2))
            gb = (gb**2).mean(0)
            if 'exact_fb' not in state:
                state['exact_fb'] = gw
            else:
                state['exact_fb'] += gw

    def _update_block(self, group, state):
        """stores covariances"""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']

        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                x = F.conv2d(x, group['gathering_filter'],
                             stride=mod.stride, padding=mod.padding,
                             groups=mod.in_channels)
            #x = x.data.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            x = x.data.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        xxt = torch.mm(x, x.t()) / float(x.shape[1])
        if 'xxt' not in state:
            state['xxt'] = xxt
        else:
            state['xxt'] += xxt

        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            #gy = gy.contiguous().view(gy.shape[0], -1)
            gy = gy.reshape(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1

        ggt = torch.mm(gy, gy.t()) / float(gy.shape[1])
        if 'ggt' not in state:
            state['ggt'] = ggt
        else:
            state['ggt'] += ggt

    def _update_block_moments(self, group, state, g=None, gb=None):
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

        if 'm2' not in state:
            state['m2'] = (g_kfe.detach()**2).sum(0)
        else:
            state['m2'] += (g_kfe.detach()**2).sum(0)

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
