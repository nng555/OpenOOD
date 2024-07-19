from typing import Any
import time
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
        self.fisher_temp = self.args.fisher_temp
        self.temp = self.args.temp
        self.nsamples = self.args.nsamples
        self.variance = self.args.variance
        self.only_logit = self.args.only_logit
        self.ce_left = self.args.ce_left
        self.scale_logit = self.args.scale_logit

        if 'swag_path' in self.args and self.args.swag_path != 'default':
            self.swag = torch.load(self.args.swag_path)
        else:
            self.swag = None

        self.nsteps = self.args.nsteps
        self.natural = self.args.natural

        self.regret = self.args.regret
        if self.args.fisher_regret == 'default':
            self.fisher_regret = self.regret
        else:
            self.fisher_regret = self.args.fisher_regret

        self.phase = self.args.phase

        self.all_classes = self.args.all_classes
        self.loss_scale = self.args.loss_scale
        self.sum_labels = self.args.sum_labels
        self.total_eps = self.args.total_eps
        self.sua = self.args.sua
        self.state_path = self.args.state_path
        self.jac_chunk_size = self.args.jac_chunk_size
        self.nsteps = self.args.nsteps
        self.empirical = self.args.empirical
        self.argmax = self.args.argmax
        self.ind_sample = self.args.ind_sample
        self.moments_chunk_size = self.args.moments_chunk_size
        self.topk = self.args.topk
        if self.jac_chunk_size == -1:
            self.jac_chunk_size = None
        if self.moments_chunk_size == -1:
            self.moments_chunk_size = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        print(net)

        self.optimizer = EKFAC(
            net,
            eps=self.damping,
            sua=self.args.sua,
            layer_eps=self.layer_eps,
            eps_type=self.eps_type,
        )

        self.ntrain = len(id_loader_dict['train'].dataset)

        if not self.natural or self.swag is not None:
            self.setup_flag = True

        if not self.setup_flag:
            net.eval()

            # running stats messes up vmap
            if self.state_path != "None" and os.path.exists(self.state_path):
                print(f"Loading EKFAC state from {self.state_path}")
                self.optimizer.load_state_dict(torch.load(self.state_path))
            else:
                self.numel = np.sum([p.numel() for p in net.parameters()])
                self.optimizer.init_hooks(net)

                # update over full dataset if not specified
                if self.eigen_iter == -1:
                    # could probably use the last batch but then the equal weighting is thrown off
                    self.eigen_iter = len(id_loader_dict['train'])

                if self.moments_iter == -1:
                    self.moments_iter = len(id_loader_dict['train'])

                # accumulate activations and gradients
                eigen_counter = 0

                with torch.enable_grad():
                    for i, batch in enumerate(tqdm(id_loader_dict['train'],
                                                   desc="EKFAC State: ",
                                                   position=0,
                                                   leave=True,
                                                   total=self.eigen_iter)):

                        if i == self.eigen_iter:
                            break

                        data = batch['data'].cuda()
                        eigen_counter += len(data)

                        net.zero_grad()
                        self.optimizer.zero_grad()

                        logits = net(data)
                        if self.topk != -1:
                            assert self.empirical is not True, "Can't ensure true label is in topk"
                            logits = torch.topk(logits, self.topk, -1)[0]
                        probs = F.softmax(logits.detach() / self.fisher_temp, -1)

                        for i in range(self.nsamples):
                            # sample a single class
                            if self.empirical:
                                labels = batch['label'].cuda()
                            elif self.argmax:
                                labels = probs.detach().argmax(-1)
                            else:
                                if self.fisher_regret == 'logit':
                                    labels = logits.detach() + torch.randn(logits.shape).cuda() * self.variance
                                #labels = torch.tensor(np.random.choice(self.num_classes, len(data))).long().cuda()
                                else:
                                    labels = torch.multinomial(probs.detach(), 1).squeeze()

                            if self.fisher_regret == 'logit':
                                #loss = (logits - (logits * F.one_hot(labels)).sum(-1).unsqueeze(-1)).mean(-1).sum()
                                #if self.empirical or self.argmax:
                                #loss = (logits * F.one_hot(labels, self.num_classes)).sum(-1).mean(-1).sum()
                                #else:
                                loss = 0.5 * ((logits - labels)**2).mean(-1).sum()
                            else:
                                loss = F.cross_entropy(logits / self.fisher_temp, labels, reduction='mean')
                            loss.backward(retain_graph=(i != self.nsamples - 1))
                            self.optimizer.update_state()
                            eigen_counter += len(data)

                # update eigenbasis
                self.optimizer.average_state(eigen_counter)
                self.optimizer.compute_basis()

                # clean up optimizer
                self.optimizer.clear_hooks()
                self.optimizer.clear_cache()
                self.optimizer.zero_grad()
                net.zero_grad()

                # update moments per example
                params = {k: v.detach() for k, v in net.named_parameters()}
                buffers = {k: v.detach() for k, v in net.named_buffers()}

                def moments_single(idv_ex, target):
                    if self.fisher_regret == 'logit':
                        def loss(p, b, d):
                            logits = functional_call(net, (p, b), (d,))
                            #if self.empirical or self.argmax:
                            #loss = (logits * target).sum(-1).mean(-1).sum()
                            #else:
                            #    loss = 0.5 * ((logits - target)**2).mean(-1).sum()
                            loss = 0.5 * ((logits - target)**2).mean(-1).sum()
                            return loss

                        grads = grad(loss)(params, buffers, idv_ex.unsqueeze(0))
                    else:
                        label = F.one_hot(target, self.num_classes)
                        grads = grad(lambda p, b, d: (-F.log_softmax(functional_call(net, (p, b), (d,)) / self.fisher_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
                    return grads

                vmap_moments = vmap(
                    moments_single,
                    in_dims=(0, 0),
                    randomness='different',
                    chunk_size=self.moments_chunk_size,
                )

                # fit second moments in eigenbasis
                moments_counter = 0
                for i, batch in enumerate(tqdm(id_loader_dict['train'],
                                               desc="EKFAC Moments: ",
                                               position=0,
                                               leave=True,
                                               total=self.moments_iter)):
                    if i == self.moments_iter:
                        break
                    data = batch['data'].cuda()

                    logits = net(data)
                    if self.topk != -1:
                        assert self.empirical is not True, "Can't ensure true label is in topk"
                        logits = torch.topk(logits, self.topk, -1)[0]
                    probs = F.softmax(logits.detach() / self.fisher_temp, -1)

                    for _ in range(self.nsamples):
                        # sample a label
                        if self.empirical:
                            targets = F.one_hot(batch['label'].cuda(), self.num_classes)
                        elif self.argmax:
                            targets = probs.argmax(-1)
                        else:
                            if self.fisher_regret == 'logit':
                                #targets = torch.tensor(np.random.choice(self.num_classes, len(data))).long().cuda()
                                targets = logits.detach() + torch.randn(logits.shape).cuda() * self.variance
                            else:
                                targets = torch.multinomial(probs.detach(), 1).squeeze()

                        grads = vmap_moments(data, targets)
                        grads = {v: grads[k] for k, v in net.named_parameters()}
                        self.optimizer.update_moments(grads)
                        moments_counter += len(data)

                self.optimizer.average_moments(moments_counter)
                self.optimizer.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()
                net.zero_grad()
                self.optimizer.zero_grad()

                if self.state_path != "None":
                    torch.save(self.optimizer.state_dict(), self.state_path)

            self.optimizer.print_eigens()
            self.setup_flag = True

            if self.phase:
                """
                orig_outs = []
                for batch in tqdm(id_loader_dict['train'], total=self.moments_iter):
                    orig_outs.append(F.softmax(net(batch['data'].cuda()) / self.fisher_temp, -1))
                orig_outs = torch.concatenate(orig_outs)
                """
                params = {k: v.detach() for k, v in net.named_parameters()}
                buffers = {k: v.detach() for k, v in net.named_buffers()}

                for test_batch in id_loader_dict['test']:
                    break
                data = test_batch['data'][0].cuda()
                label = F.one_hot(torch.tensor([1]), self.num_classes).cuda()

                orig_test_out = F.softmax(net(data.unsqueeze(0)) / self.fisher_temp)

                grads = grad(lambda p, b, d: (-F.log_softmax(functional_call(net, (p, b), (d,)) / self.fisher_temp, -1) * label).sum())(params, buffers, data.unsqueeze(0))
                grads = {v: [k, grads[k].unsqueeze(0)] for k, v in net.named_parameters()}

                norm, nat_grads = self.optimizer.step(grads=grads, return_grads=True)

                for i in range(0, 100):
                    new_params = {k: -1e-7 * v[0] + params[k] for k, v in nat_grads.items()}
                    """
                    new_outs = []
                    for batch in tqdm(id_loader_dict['train'], total=self.moments_iter):
                        new_outs.append(F.softmax(functional_call(net, (new_params, buffers), (batch['data'].cuda(),)) / self.fisher_temp, -1))
                    new_outs = torch.concatenate(new_outs)
                    """
                    new_test_out = F.softmax(functional_call(net, (new_params, buffers), (data.unsqueeze(0),)) / self.fisher_temp)


        else:
            pass


    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, labels=None):
        data = data.cuda()

        params = {k: v.detach() for k, v in net.named_parameters()}
        buffers = {k: v.detach() for k, v in net.named_buffers()}

        timer = time.time()

        logits, features = net.forward(data, return_feature=True)
        pred = torch.argmax(logits, -1)

        def grad_mislabel_fn(idv_ex, new_target, erase_target):
            new_label = F.one_hot(new_target, self.num_classes)
            erase_label = F.one_hot(erase_target, self.num_classes)
            erase_label = new_label - erase_label

            def loss(p, b, d, erase=False):
                logits = functional_call(net, (p, b), (d,))
                if self.topk != -1:
                    logits = torch.topk(logits, self.topk, -1)[0]
                if not erase:
                    loss = (-F.log_softmax(logits / self.temp, -1) * new_label).sum()
                else:
                    loss = (-F.log_softmax(logits / self.temp, -1) * erase_label).sum()
                return loss

            new_grads = grad(loss)(params, buffers, idv_ex.unsqueeze(0), True)
            new_grads = {v: [k, new_grads[k].unsqueeze(0)] for k, v in net.named_parameters()}
            _, nat_grads = self.optimizer.step(grads=new_grads, return_grads=True)
            grads = grad(loss)(params, buffers, idv_ex.unsqueeze(0), False)
            norm = self.optimizer.dict_dot(grads, nat_grads)
            return norm

        def grad_step_fn(idv_ex, target, label):
            label = F.one_hot(target, self.num_classes)

            def loss(p, b, d):
                logits = functional_call(net, (p, b), (d,))
                if self.topk != -1:
                    logits = torch.topk(logits, self.topk, -1)[0]
                loss = (-F.log_softmax(logits, -1) * label).sum()
                if self.swag is not None:
                    loss = 0.5 * loss
                    for k in params:
                        loss = loss + ((p[k] - params[k])**2).sum()
                return loss

            new_params = params.copy()

            if self.natural:
                for _ in range(self.nsteps):
                    grads = grad(loss)(new_params, buffers, idv_ex.unsqueeze(0))
                    if self.swag is not None:
                        for k in grads:
                            grads[k] = grads[k] * self.swag[k]
                        new_params = {k: params[k] - 0.5 * grads[k] for k in grads}
                    else:
                        grads = {v: [k, grads[k].unsqueeze(0)] for k, v in net.named_parameters()}
                        norm, nat_grads = self.optimizer.step(grads=grads, return_grads=True)
                        new_params = {k: - 0.0001 * (1 / self.ntrain) * v[0] + params[k] for k, v in nat_grads.items()}
            else:
                for _ in range(self.nsteps):
                    grads = grad(loss)(new_params, buffers, idv_ex.unsqueeze(0))
                    new_params = {k: -0.0001 * (1 / self.ntrain) * v + params[k] for k, v in grads.items()}

            new_out = F.softmax(functional_call(net, (new_params, buffers), (idv_ex.unsqueeze(0),)), -1)
            return (new_out[0] * label).sum()

        # helper function for vmapping
        def grad_lin_fn(idv_ex, target, label):
            if self.topk == -1:
                label = F.one_hot(target, self.num_classes)
            else:
                label = F.one_hot(target, self.topk)

            def loss(p, b, d):
                logits = functional_call(net, (p, b), (d,))
                if self.topk != -1:
                    logits = torch.topk(logits, self.topk, -1)[0]
                loss = (-F.log_softmax(logits / self.temp, -1) * label).sum()
                return loss

            grads = grad(loss)(params, buffers, idv_ex.unsqueeze(0))

            if self.natural:
                grads = {v: [k, grads[k].unsqueeze(0)] for k, v in net.named_parameters()}
                norm = self.optimizer.step(grads=grads)
                return norm
                #norm, nat_grads = self.optimizer.step(grads=grads, return_grads=True)
                #nat_norm = self.optimizer.dict_dot(nat_grads, nat_grads)
                #return norm, nat_norm
            else:
                norm = self.optimizer.dict_dot(grads, grads)
            return norm

            #norm, nat_grads = self.optimizer.step(grads=grads)
            #self_nak = self.optimizer.dict_dot(grads, nat_grads)
            #return norm, self_nak

        def grad_logit_fn(idv_ex, target, label):
            if self.topk == -1:
                label = F.one_hot(target, self.num_classes)
            else:
                label = F.one_hot(target, self.topk)

            def ce_loss(p, b, d):
                logits = functional_call(net, (p, b), (d,))
                if self.topk != -1:
                    logits = torch.topk(logits, self.topk, -1)[0]
                loss = (-F.log_softmax(logits / self.temp, -1) * label).sum()
                return loss

            def logit_loss(p, b, d):
                logits = functional_call(net, (p, b), (d,))
                if self.topk != -1:
                    logits = torch.topk(logits, self.topk, -1)[0]
                if self.only_logit:
                    loss = -(logits * label).sum()
                else:
                    loss = logits - (logits * label).sum(-1).unsqueeze(-1)
                    loss = loss.mean(-1).sum()
                return loss

            grads = grad(logit_loss)(params, buffers, idv_ex.unsqueeze(0))

            if self.natural:
                if self.ce_left:
                    grads = {v: [k, grads[k].unsqueeze(0)] for k, v in net.named_parameters()}
                    norm, nat_grads = self.optimizer.step(grads=grads, return_grads=True)
                    grads = grad(ce_loss)(params, buffers, idv_ex.unsqueeze(0))
                    norm = self.optimizer.dict_dot(grads, nat_grads)
                else:
                    grads = {v: [k, grads[k].unsqueeze(0)] for k, v in net.named_parameters()}
                    norm = self.optimizer.step(grads=grads)
            else:
                norm = self.optimizer.dict_dot(grads, grads)
            return norm

        if self.regret == 'linear':
            grad_fn = grad_lin_fn
        elif self.regret == 'step':
            grad_fn = grad_step_fn
        elif self.regret == 'mislabel':
            grad_fn = grad_mislabel_fn
        elif self.regret == 'logit':
            grad_fn = grad_logit_fn

        def process_single_grad(idv_ex, label):
            if self.topk == -1:
                return vmap(grad_fn, in_dims=(None, 0, None))(idv_ex, torch.arange(self.num_classes).cuda(), label)
            else:
                return vmap(grad_fn, in_dims=(None, 0, None))(idv_ex, torch.arange(self.topk).cuda(), label)

        """
        def process_single_jac(idv_ex):
            fjac = jacrev(lambda p, b, d: -F.log_softmax(functional_call(net, (p, b), (d,)) / self.temp, -1))(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: fjac[k][0] for k, v in net.named_parameters()}
            nat_grads = self.optimizer.step(grads=grads)
            self_nak = self.optimizer.dict_bdot(grads, nat_grads)
            return self_nak.diagonal()

        vmap_jac = vmap(process_single_jac, in_dims=(0,), chunk_size=self.jac_chunk_size)
        """

        if labels is None:
            label_dim = None
        else:
            label_dim = 0

        if not self.all_classes and labels is not None:
            vmap_grad = vmap(grad_fn, in_dims=(0, label_dim, label_dim), chunk_size=self.jac_chunk_size)
            norm = vmap_grad(data, labels, labels)
        else:
            vmap_grad = vmap(process_single_grad, in_dims=(0, label_dim,), chunk_size=self.jac_chunk_size)
            norm = vmap_grad(data, labels)

        norm = norm.squeeze()
        ptime = time.time() - timer

        if self.topk != -1:
            logits = torch.topk(logits, self.topk, -1)[0]

        probs = F.softmax(logits / self.temp, -1)
        if self.regret == 'step':
            regret = torch.log(norm) - torch.log(probs)
            comp = probs * torch.exp(regret / self.temp)
            comp = comp.sum(-1)
            conf = -comp
            return pred, conf, {'norm': norm.cpu().numpy(), 'logits': logits.detach().cpu().numpy(), 'raw_probs': F.softmax(logits, -1).detach().cpu().numpy(), 'probs': F.softmax(logits / self.temp, -1).detach().cpu().numpy(), 'regret': regret.cpu().numpy(), 'comp': comp.cpu().numpy(), 'ptime': ptime}
        else:
            #if self.regret == 'logit':
            #    pnorm = norm * (1 - probs)
            #else:
            #pnorm = norm
            #p_norm = torch.log(1 + norm / self.ntrain)
            #import ipdb; ipdb.set_trace()
            #raw_conf = torch.exp(p_norm / (self.temp))

            if self.all_classes:
                if self.scale_logit:
                    conf = -((1 - probs) * probs * norm).sum(-1)
                else:
                    #conf = -(probs * norm).sum(-1)
                    conf = -(probs * torch.log(probs * norm)).sum(-1)

            else:
                conf = -norm

            #conf = -norm.mean(-1) + 0.1 * ((norm**2).mean(-1) - norm.mean(-1)**2)
            #norm = probs * norm
            #conf = ((norm**2).mean(-1) - norm.mean(-1)**2)
            if labels is not None:
                error_norm = ((probs - F.one_hot(labels, self.num_classes))**2).detach().cpu().numpy().sum(-1)
                if self.all_classes:
                    grand = np.take_along_axis(norm.cpu().numpy(), labels.cpu().numpy()[:, None], -1)[:, 0]
                else:
                    grand = norm.cpu().numpy()
            else:
                error_norm = None
                grand = None

            #nml = F.softmax(logits / self.temp, -1)
            #nml = nml + nml * norm * self.alpha / 50000
            #nml /= nml.sum(-1)[:, None]

            res = {'grand': grand, 'error_norm': error_norm, 'norm': norm.cpu().numpy(), 'logits': logits.detach().cpu().numpy(), 'raw_probs': F.softmax(logits, -1).detach().cpu().numpy(), 'probs': F.softmax(logits / self.temp, -1).detach().cpu().numpy(), 'ptime': ptime}

            res = {k: v for k, v in res.items() if v is not None}

            return pred, conf, res

    def set_hyperparam(self, hyperparam: list):
        self.temp = hyperparam[0]
        self.damping = hyperparam[1]

    def get_hyperparam(self):
        return [self.temp, self.damping]
