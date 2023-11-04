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
from openood.preprocessors.transform import normalization_dict
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
        self.fuse = self.args.fuse
        self.double = self.args.double
        self.sua = self.args.sua
        self.eigen_iter = self.args.eigen_iter
        self.moments_iter = self.args.moments_iter
        self.left_output = self.args.left_output
        self.right_output = self.args.right_output
        self.ntrain = self.args.ntrain
        self.featskip = self.args.featskip
        self.ce_weight = self.args.ce_weight
        self.nfeature_reduce = self.args.nfeature_reduce

        # ODIN
        self.odin = self.args.odin
        self.noise = self.args.noise
        try:
            self.input_std = normalization_dict[self.config.dataset.name][1]
        except KeyError:
            self.input_std = [0.5, 0.5, 0.5]

        self.top_layer = self.args.top_layer
        self.layer_eps = self.args.layer_eps
        self.eps_type = self.args.eps_type
        self.remove_top = self.args.remove_top

        if self.args.temp == -1:
            self.loss_temp = self.args.loss_temp
            self.sample_temp = self.args.sample_temp
            self.fsample_temp = self.args.fsample_temp
            self.floss_temp = self.args.floss_temp
        else:
            self.loss_temp = self.sample_temp = self.fsample_temp = self.floss_temp = self.args.temp

        self.inverse = self.args.inverse
        self.per_class_m1 = self.args.per_class_m1
        self.floss_fn = self.args.floss_fn
        self.all_classes = self.args.all_classes
        self.sum_labels = self.args.sum_labels
        self.off_diag = self.args.off_diag
        self.eigenscale = self.args.eigenscale
        self.layerscale = self.args.layerscale
        self.total_eps = self.args.total_eps
        self.center = self.args.center
        self.state_path = self.args.state_path + '.cpt'
        self.grad_path = self.args.state_path + '_grad.cpt'
        self.spec_path = self.args.state_path + '_spec.cpt'
        self.jac_chunk_size = self.args.jac_chunk_size
        self.empirical = self.args.empirical
        self.moments_chunk_size = self.args.moments_chunk_size

        # just assume we're going to calculate over all classes, since its a one time cost
        self.topk = self.args.topk

        if self.jac_chunk_size == -1:
            self.jac_chunk_size = None
        if self.moments_chunk_size == -1:
            self.moments_chunk_size = None

    def setup_means(self, net, loader):
        net.zero_grad()
        self.optimizer.zero_grad()
        total_weight = torch.zeros(self.num_classes).cuda()

        for i, batch in enumerate(tqdm(loader,
                                       desc="Means: ",
                                       position=0,
                                       leave=True,
                                       total=self.eigen_iter)):
            if i == self.eigen_iter:
                break

            data = batch['data'].cuda()
            logits = net(data)

            if self.fsample_temp == -1:
                probs = F.softmax(torch.ones_like(logits), -1)
            else:
                probs = F.softmax(logits / self.fsample_temp, -1).detach()

            for kcls in range(self.num_classes):
                labels = torch.ones(len(logits)).cuda().long() * kcls
                loss = F.cross_entropy(logits / self.floss_temp, labels)
                loss.backward(retain_graph=(kcls != self.num_classes - 1))
                self.optimizer.update_mean(probs[:, kcls], labels)
                net.zero_grad()
                self.optimizer.zero_grad()

            total_weight += probs.sum(0)

        self.optimizer.average_means(total_weight)

    def setup_avg_grad(self, net, loader):
        net.zero_grad()
        self.optimizer.zero_grad()

        avg_grads = [None for _ in range(self.num_classes)]

        for i, batch in enumerate(tqdm(loader,
                                       desc='Average Grad: ',
                                       position=0,
                                       leave=True,
                                       total=self.eigen_iter)):
            if i == self.eigen_iter:
                break

            data = batch['data'].cuda()
            logits = net(data)

            if self.fsample_temp == -1:
                probs = F.softmax(torch.ones_like(logits), -1)
            else:
                probs = F.softmax(logits / self.fsample_temp, -1).detach()

            for kcls in range(self.num_classes):
                net.zero_grad()
                self.optimizer.zero_grad()
                labels = torch.ones(len(logits)).cuda().long() * kcls
                loss = self.floss_temp**2 * F.cross_entropy(logits / self.floss_temp, labels, reduction='sum') / self.ntrain
                loss.backward(retain_graph=(kcls != self.num_classes - 1))
                grads = {k: v.grad.data for k, v in net.named_parameters()}
                if avg_grads[kcls] is None:
                    avg_grads[kcls] = grads
                else:
                    avg_grads[kcls] = {k: v + avg_grads[kcls][k] for k, v in grads.items()}

        net.zero_grad()
        self.optimizer.zero_grad()
        return avg_grads

    def setup_raw_moments(self, net, loader):
        if self.per_class_m1:
            total_weight = torch.zeros((self.num_classes)).cuda()
        else:
            total_weight = 0

        for i, batch in enumerate(tqdm(loader,
                                       desc="Raw Moments: ",
                                       position=0,
                                       leave=True,
                                       total=self.eigen_iter)):
            if i == self.eigen_iter:
                break

            data = batch['data'].cuda()
            if self.double:
                data = data.double()
            true_labels = batch['label'].cuda()

            net.zero_grad()
            self.optimizer.zero_grad()

            logits = net(data)

            if self.fsample_temp == -1:
                probs = F.softmax(torch.ones_like(logits), -1)
            else:
                probs = F.softmax(logits / self.fsample_temp, -1).detach()

            raw_probs = F.softmax(logits, -1).detach()

            if self.topk == -1:
                net.zero_grad()
                self.optimizer.zero_grad()
                labels = torch.multinomial(probs, 1).squeeze()
                loss = self.floss_temp**2 * F.cross_entropy(logits / self.floss_temp, labels)
                loss.backward()
                self.optimizer.update_state(labels=labels)
                if self.per_class_m1:
                    for kcls in range(self.num_classes):
                        total_weight[kcls] += (labels == kcls).sum()
            else:
                # sample all classes and weight
                for kcls in range(self.num_classes):

                    net.zero_grad()
                    self.optimizer.zero_grad()
                    labels = torch.ones(len(logits)).cuda().long() * kcls
                    loss = self.floss_temp**2 * F.cross_entropy(logits / self.floss_temp, labels)
                    loss.backward(retain_graph=(kcls != self.num_classes - 1))

                    # centering happens inside second moment estimator
                    # weight by probabilities
                    self.optimizer.update_state(ex_weight=probs[:, kcls], labels=labels)
                    if self.per_class_m1:
                        total_weight[kcls] += probs[:, kcls].sum()

            if not self.per_class_m1:
                total_weight += len(logits)

        self.optimizer.average_state(total_weight)

    def setup_eigen_moments(self, net, loader, moment=2):
        params = {k: v.detach() for k, v in net.named_parameters()}
        buffers = {k: v.detach() for k, v in net.named_buffers()}

        def moments_single_ce(idv_ex, target):
            label = F.one_hot(target, self.num_classes)
            grads = grad(lambda p, b, d: (-self.floss_temp**2 * F.log_softmax(functional_call(net, (p, b), (d,)) / self.floss_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
            return grads

        def moments_single_logits(idv_ex, target):
            # use a multiplicative mask inside vmap
            label = F.one_hot(target, self.num_classes)
            def logits_loss(params, buffers, data):
                out = functional_call(net, (params, buffers), (data,)) / self.floss_temp
                loss = 0.5 * (out**2 * label).sum() + (out * label).sum() * out.sum()
                return loss
            grads = grad(lambda p, b, d: logits_loss(p, b, d))(params, buffers, idv_ex.unsqueeze(0))
            return grads

        def moments_single_distill(idv_ex, raw_target, temp_target):
            label = F.one_hot(raw_target, self.num_classes)
            temp_label = F.one_hot(temp_target, self.num_classes)
            def loss(params, buffers, data):
                out = functional_call(net, (params, buffers), (data,))
                base_loss = (-F.log_softmax(out) * label).sum()
                temp_loss = (-F.log_softmax(out / self.floss_temp) * temp_label).sum()
                return base_loss + temp_loss * self.floss_temp**2
            grads = grad(lambda p, b, d: loss(p, b, d))(params, buffers, idv_ex.unsqueeze(0))
            return grads

        if self.floss_fn == 'ce':
            moment_fn = moments_single_ce
        elif self.floss_fn == 'logits':
            moment_fn = moments_single_logits
        elif self.floss_fn == 'distill':
            moment_fn = moments_single_distill

        vmap_moments = vmap(
            moment_fn,
            in_dims=(0, 0),
            randomness='different',
            chunk_size=self.moments_chunk_size,
        )

        # fit second moments in eigenbasis
        if self.per_class_m1:
            total_weight = torch.zeros((self.num_classes)).cuda()
        else:
            total_weight = 0
        for i, batch in enumerate(tqdm(loader,
                                       desc="Eigen Moments: ",
                                       position=0,
                                       leave=True,
                                       total=self.moments_iter)):
            if i == self.moments_iter:
                break

            data = batch['data'].cuda()
            if self.double:
                data = data.double()
            logits = net(data)

            if self.fsample_temp == -1:
                probs = F.softmax(torch.ones_like(logits), -1)
            else:
                probs = F.softmax(logits / self.fsample_temp, -1).detach()

            raw_probs = F.softmax(logits, -1).detach()

            if self.topk == -1:
                labels = torch.multinomial(probs, 1).squeeze()
                grads = vmap_moments(data, labels)
                grads = {v: grads[k] for k, v in net.named_parameters()}
                self.optimizer.update_moments(grads=grads, labels=labels)
                del grads
                if self.per_class_m1:
                    for kcls in range(self.num_classes):
                        total_weight[kcls] += (labels == kcls).sum()
            else:
                for kcls in range(self.num_classes):
                    labels = torch.ones(len(data)).cuda().long() * kcls
                    grads = vmap_moments(data, labels)
                    # center gradients before projection
                    if self.center:
                        grads = {v: grads[k] - self.avg_grad[k][kcls] for k, v in net.named_parameters()}
                    else:
                        grads = {v: grads[k] for k, v in net.named_parameters()}

                    self.optimizer.update_moments(grads=grads, ex_weight=probs[:, kcls], labels=labels)
                    del grads
                    if self.per_class_m1:
                        total_weight[kcls] += probs[:, kcls].sum()

            if not self.per_class_m1:
                total_weight += len(data)

        self.optimizer.average_moments(total_weight)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        if self.double:
            net.to(torch.double)

        if self.fuse:
            net.fuse_conv_bn_layers()

        net.eval()

        self.optimizer = EKFAC(
            net,
            eps=self.damping,
            num_classes=self.num_classes,
            sua=self.sua,
            layer_eps=self.layer_eps,
            eps_type=self.eps_type,
            eigenscale=self.eigenscale,
            featskip=self.featskip,
            layerscale=self.layerscale,
            center=self.center,
            per_class_m1=self.per_class_m1,
            nfeature_reduce=self.nfeature_reduce,
        )

        # maybe need to modify this depending on whether we use the full dataset
        self.ntrain = len(id_loader_dict['train'].sampler)

        if self.eigen_iter == -1:
            self.eigen_iter = len(id_loader_dict['train'])

        if self.moments_iter == -1:
            self.moments_iter = len(id_loader_dict['train'])

        if not self.setup_flag:
            # setup average grad for centering
            if self.center:
                if os.path.exists(self.grad_path):
                    print(f"Loading grad state from {self.grad_path}")
                    self.avg_grad = torch.load(self.grad_path)
                else:
                    self.avg_grad = self.setup_avg_grad(net, id_loader_dict['train'])
                    self.avg_grad = {k: torch.stack([g[k] for g in self.avg_grad]) for k in self.avg_grad[0]}
                    torch.save(self.avg_grad, self.grad_path)

                avg_grad_norm = self.optimizer.dict_bdot(self.avg_grad, self.avg_grad).diagonal()
                print(f"Average grad l2 norm: {avg_grad_norm}")

            if self.state_path != "None.cpt" and os.path.exists(self.state_path):
                print(f"Loading EKFAC state from {self.state_path}")
                self.optimizer.load_state_dict(torch.load(self.state_path))
            else:
                self.numel = np.sum([p.numel() for p in net.parameters()])

                with torch.enable_grad():

                    self.optimizer.init_hooks(net)

                    # accumulate means for centering
                    if self.center:
                        self.setup_means(net, id_loader_dict['train'])

                    # accumulate raw second moments
                    self.setup_raw_moments(net, id_loader_dict['train'])

                    # find eigenbasis
                    self.optimizer.compute_basis()

                    # clean up optimizer
                    self.optimizer.clear_hooks()
                    self.optimizer.clear_cache()
                    self.optimizer.zero_grad()
                    net.zero_grad()

                    # reaccumulate eigenbasis second moments
                    self.setup_eigen_moments(net, id_loader_dict['train'])

                    # clean up again
                    gc.collect()
                    torch.cuda.empty_cache()
                    net.zero_grad()
                    self.optimizer.zero_grad()
                    torch.save(self.optimizer.state_dict(), self.state_path)

            if not os.path.exists(self.spec_path):
                torch.save(self.optimizer.get_spectrum(), self.spec_path)

            self.optimizer.print_eigens()
            self.optimizer.get_spectrum()

            # remove top C^2 eigenvalues
            if self.remove_top:
                self.optimizer.remove_top_moments(self.num_classes**2)

            self.setup_flag = True

            # change batchnorm layers to training mode?
            #for mod in net.modules():
            #    mod_class = mod.__class__.__name__
            #    if mod_class == 'BatchNorm2d':
            #        mod.track_running_stats=False
            #        mod.train()
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """
        data.requires_grad = True
        output = net(data)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(output, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
        gradient[:, 1] = (gradient[:, 1]) / self.input_std[1]
        gradient[:, 2] = (gradient[:, 2]) / self.input_std[2]

        # Adding small perturbations to images
        tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)
        """
        data = data.cuda()
        if self.double:
            data = data.double()
        logits = net(data)
        pred = torch.argmax(logits, -1)

        params = {k: v.detach() for k, v in net.named_parameters()}
        #buffers = {k: v.detach() for k, v in net.named_buffers()}
        buffers = {}

        def grad_single_loss_center(idv_ex, target, avg_grad):
            label = F.one_hot(target, self.num_classes)

            # calculate and center gradients
            grads = grad(lambda p, b, d: (-F.log_softmax(functional_call(net, (p, b), (d,)) / self.loss_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: (grads[k] - avg_grad[k]).unsqueeze(0) for k, v in net.named_parameters()}

            nat_grads, norm = self.optimizer.step(grads=grads, inverse=self.inverse, l2=True, labels=target)
            self_nak = self.optimizer.dict_dot(grads, nat_grads, return_layers=True)
            ntk = self.optimizer.dict_dot(grads, grads, return_layers=True)
            return self_nak, ntk, norm

        def grad_single_loss(idv_ex, target):
            label = F.one_hot(target, self.num_classes)

            # calculate and center gradients
            grads = grad(lambda p, b, d: (-F.log_softmax(functional_call(net, (p, b), (d,)) / self.loss_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: grads[k].unsqueeze(0) for k, v in net.named_parameters()}

            _, norm, feat = self.optimizer.step(grads=grads, inverse=self.inverse, l2=True, labels=target, return_feats=True)
            return norm, feat
            #self_nak = self.optimizer.dict_dot(grads, nat_grads, return_layers=True)
            #ntk = self.optimizer.dict_dot(grads, grads, return_layers=True)
            #return self_nak, ntk, norm

        def grad_single_loss_distill(idv_ex, target, temp_target):
            label = F.one_hot(target, self.num_classes)
            temp_label = F.one_hot(temp_target, self.num_classes)
            def loss(params, buffers, data):
                out = functional_call(net, (params, buffers), (data,))
                base_loss = (-F.log_softmax(out) * label).sum()
                temp_loss = (-F.log_softmax(out / self.loss_temp) * temp_label).sum()
                return base_loss + temp_loss * self.loss_temp**2
            grads = grad(lambda p, b, d: loss(p, b, d).sum())(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: grads[k].unsqueeze(0) for k, v in net.named_parameters()}
            nat_grads = self.optimizer.step(grads=grads, inverse=self.inverse, l2=False)
            naks = self.optimizer.dict_dot(grads, nat_grads, return_layers=True)
            ntks = self.optimizer.dict_dot(grads, grads, return_layers=True)
            return naks, ntks

        def grad_single_loss_with_feat(idv_ex, target):
            label = F.one_hot(target, self.num_classes)
            grads = grad(lambda p, b, d: (-F.log_softmax(functional_call(net, (p, b), (d,)) / self.loss_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: grads[k].unsqueeze(0) for k, v in net.named_parameters()}
            self_nak, eigenfeat = self.optimizer.step(grads=grads, inverse=self.inverse, l2=True, return_feats=True)
            return self_nak, eigenfeat

        if self.center:
            grad_fn = grad_single_loss_center
        else:
            grad_fn = grad_single_loss

        if self.center:
            def process_single_grad(idv_ex, idv_logits):
                res = vmap(grad_fn, in_dims=(None, 0, 0))(idv_ex, torch.arange(self.num_classes).cuda(), self.avg_grad)
                return res
        else:
            def process_single_grad(idv_ex, idv_logits):
                res = vmap(grad_fn, in_dims=(None, 0))(idv_ex, torch.arange(self.num_classes).cuda())
                return res

        vmap_grad = vmap(process_single_grad, in_dims=(0, 0,), chunk_size=self.jac_chunk_size, randomness='different')
        norm, feats = vmap_grad(data, logits) # N x K x L
        #nak = torch.stack(nak, -1)
        #cum_nak = nak.sum(-1)

        """
        full_nak = []
        feat_accum = None

        for idv_ex, idv_logits in zip(data, logits):
            nak, eigenfeat = vmap(grad_single_loss_with_feat, in_dims=(None, 0))(idv_ex, torch.arange(self.num_classes).cuda())
            full_nak.append(nak)
            eigenfeat = torch.sum(eigenfeat * F.softmax(idv_logits / self.sample_temp, -1)[:, None], 0)
            if feat_accum is None:
                feat_accum = eigenfeat
            else:
                feat_accum += eigenfeat
        nak = torch.stack(full_nak)
        """

        probs = F.softmax(logits / self.sample_temp, -1)

        if self.sample_temp == -1:
            conf = torch.mean(-norm, -1)
        else:
            conf = torch.sum(-norm * probs, -1)
            feats = torch.sum(feats * probs[..., None], 1)

        return pred, conf, {'conf': conf, 'norm': norm, 'logits': logits, 'eigenfeat': feats, 'probs': probs}
        #return pred, conf, {'self_nak': nak, 'conf': conf, 'norm': norm, 'logits': logits, 'ntk': torch.stack(ntk, -1), 'probs': probs}
        #return pred, conf, {'self_nak': nak, 'logits': logits, 'eigenfeat': feat_accum}

