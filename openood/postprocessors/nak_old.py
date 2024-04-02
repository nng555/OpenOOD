from typing import Any
import numpy as np
import gc
import os

from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import pickle as pkl
from torch.optim.optimizer import Optimizer
from torch.func import hessian
from torch.autograd.functional import hessian as fhessian
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

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
        self.random_label = self.args.random_label
        self.fuse = self.args.fuse
        self.double = self.args.double
        self.sua = self.args.sua
        self.sud = self.args.sud
        self.whiten = self.args.whiten
        self.eigen_iter = self.args.eigen_iter
        self.moments_iter = self.args.moments_iter
        self.left_output = self.args.left_output
        self.right_output = self.args.right_output
        self.ntrain = self.args.ntrain
        self.featskip = self.args.featskip
        self.ce_weight = self.args.ce_weight
        self.nfeature_reduce = self.args.nfeature_reduce
        self.remove_top_layers = self.args.remove_top_layers
        self.remove_bot_layers = self.args.remove_bot_layers

        # ODIN
        self.odin = self.args.odin
        self.noise = self.args.noise
        self.odin_temp = self.args.odin_temp
        try:
            self.input_std = normalization_dict[self.config.dataset.name][1]
        except KeyError:
            self.input_std = [0.5, 0.5, 0.5]

        self.top_layer = self.args.top_layer
        self.layer_eps = self.args.layer_eps
        self.eps_type = self.args.eps_type
        self.remove_top = self.args.remove_top

        if self.args.temp == -1:
            self.eigen_temp = self.args.eigen_temp
            self.moments_temp = self.args.moments_temp
            self.grad_temp = self.args.grad_temp
            if self.eigen_temp == -1 and self.moments_temp == -1 and self.grad_temp == -1:
                self.temp = self.eigen_temp = self.moments_temp = self.grad_temp = 1
        else:
            self.eigen_temp = self.moments_temp = self.grad_temp = self.args.temp

        self.sandwich = self.args.sandwich
        self.inverse = not self.sandwich
        self.per_class_m1 = self.args.per_class_m1
        self.floss_fn = self.args.floss_fn
        self.all_classes = self.args.all_classes
        self.sum_labels = self.args.sum_labels
        self.off_diag = self.args.off_diag
        self.eigenscale = self.args.eigenscale
        self.layerscale = self.args.layerscale
        self.total_eps = self.args.total_eps
        self.state_path = self.args.state_path + '.cpt'
        self.grad_path = self.args.state_path + '_grad.cpt'
        self.spec_path = self.args.state_path + '_spec.cpt'
        self.ecdf_path = self.args.state_path + '_ecdf.cpt'
        self.lr_model_path = self.args.state_path + '_lr.pkl'
        self.jac_chunk_size = self.args.jac_chunk_size
        self.empirical = self.args.empirical
        self.moments_chunk_size = self.args.moments_chunk_size
        self.class_chunk_size = self.args.class_chunk_size
        self.reset_m1 = self.args.reset_m1

        self.full_fim = self.args.full_fim

        # just assume we're going to calculate over all classes, since its a one time cost
        self.topk = self.args.topk

        if self.jac_chunk_size == -1:
            self.jac_chunk_size = None
        if self.moments_chunk_size == -1:
            self.moments_chunk_size = None
        if self.class_chunk_size == -1:
            self.class_chunk_size = None

    def setup_means(self, net, loader):
        net.zero_grad()
        self.optimizer.zero_grad()
        total_weight = 0

        for i, batch in enumerate(tqdm(loader,
                                       desc="Means: ",
                                       position=0,
                                       leave=True,
                                       total=self.eigen_iter)):
            if i == self.eigen_iter:
                break

            data = batch['data'].cuda()
            logits = net(data)

            if self.moments_temp == -1:
                probs = F.softmax(torch.ones_like(logits), -1)
            else:
                probs = F.softmax(logits / self.moments_temp, -1).detach()

            labels = torch.multinomial(probs, 1).squeeze()
            loss = F.cross_entropy(logits / self.moments_temp, labels, reduction='sum')
            loss.backward()
            self.optimizer.update_mean(labels=labels)
            net.zero_grad()
            self.optimizer.zero_grad()

            total_weight += len(data)

        return self.optimizer.average_means(total_weight)

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
                loss = F.cross_entropy(logits / self.floss_temp, labels, reduction='sum') / self.ntrain
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

        int_hess = None

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

            if self.eigen_temp == -1:
                probs = F.softmax(torch.ones_like(logits), -1)
            else:
                probs = F.softmax(logits / self.eigen_temp, -1).detach()

            raw_probs = F.softmax(logits, -1).detach()

            if self.topk == -1:
                net.zero_grad()
                self.optimizer.zero_grad()
                labels = torch.multinomial(probs, 1).squeeze()

                """
                def loss_fn(logits, labels):
                    return self.eigen_temp * F.cross_entropy(logits / self.eigen_temp, labels, reduction='sum')

                self.optimizer.prev_hess = vmap(hessian(loss_fn), in_dims=(0, 0))(logits, labels)

                int_outs = net.intermediate_forward(data, 4).detach()
                def int_loss_fn(feats, labels):
                    return self.eigen_temp * F.cross_entropy(net(feats, lindex=4) / self.eigen_temp, labels, reduction='sum')
                import ipdb; ipdb.set_trace()
                _int_hess = vmap(hessian(int_loss_fn), in_dims=(0, 0))(int_outs.unsqueeze(1), labels)

                if int_hess is None:
                    int_hess = _int_hess.sum(0)
                else:
                    int_hess += _int_hess.sum(0)
                """
                if self.floss_fn == 'ce':
                    loss = F.cross_entropy(logits / self.eigen_temp, labels, reduction='sum')
                elif self.floss_fn == 'nlogits':
                    probs = F.softmax(logits / self.eigen_temp, -1)
                    one_hot_labels = F.one_hot(labels, self.num_classes)
                    cprobs = (probs * one_hot_labels).sum(-1)
                    loss = torch.log(cprobs / (1 - cprobs)).sum()
                elif self.floss_fn == 'back_ce':
                    new_logits = torch.cat((logits, torch.zeros((len(logits), 1)).cuda()), -1)
                    new_probs = F.softmax(new_logits / self.eigen_temp, -1).detach()
                    new_labels = torch.multinomial(new_probs, 1).squeeze()
                    loss = self.eigen_temp * F.cross_entropy(new_logits / self.eigen_temp, labels, reduction='sum')
                elif self.floss_fn == 'logit_margin':
                    labels = F.one_hot(torch.randint(0, self.num_classes, (len(logits),)), self.num_classes).cuda()
                    loss = (logits - (logits * labels).sum(-1).unsqueeze(-1)).mean(-1).sum()

                loss.backward()
                self.optimizer.update_state(labels=labels)
                if self.per_class_m1:
                    for kcls in range(self.num_classes):
                        total_weight[kcls] += (labels == kcls).sum()
                self.optimizer.prev_hess = None
                self.optimizer.prev_weight = None

            else:
                # sample all classes and weight
                for kcls in range(self.num_classes):

                    net.zero_grad()
                    self.optimizer.zero_grad()
                    labels = torch.ones(len(logits)).cuda().long() * kcls
                    loss = F.cross_entropy(logits / self.eigen_temp, labels, reduction='sum')
                    loss.backward(retain_graph=(kcls != self.num_classes - 1))

                    # weight by probabilities
                    self.optimizer.update_state(ex_weight=probs[:, kcls], labels=labels)
                    if self.per_class_m1:
                        total_weight[kcls] += probs[:, kcls].sum()

            if not self.per_class_m1:
                total_weight += len(logits)

        self.optimizer.average_state(total_weight)

    def setup_eigen_moments(self, net, train_loader, val_loader):
        params = {k: v.detach() for k, v in net.named_parameters()}
        buffers = {k: v.detach() for k, v in net.named_buffers()}

        def moments_single_ce(idv_ex, target):
            label = F.one_hot(target, self.num_classes)
            grads = grad(lambda p, b, d: (-F.log_softmax(functional_call(net, (p, b), (d,)) / self.moments_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
            return grads

        def moments_single_nlogits(idv_ex, target):
            label = F.one_hot(target, self.num_classes)
            def nlogit(p, b, d):
                out = (F.softmax(functional_call(net, (p, b), (d,)) / self.moments_temp, -1) * label).sum()
                loss = torch.log(out / (1 - out))
                return loss
            grads = grad(nlogit)(params, buffers, idv_ex.unsqueeze(0))
            return grads

        def moments_single_back_ce(idv_ex, target):
            label = F.one_hot(target, self.num_classes + 1)

            def back_ce(p, b, d):
                out = functional_call(net, (p, b), (d,))
                out = torch.cat((out, torch.zeros(len(out), 1).cuda()), -1)
                loss = (-self.moments_temp * F.log_softmax(out, -1) * label).sum()
                return loss

            grads = grad(back_ce)(params, buffers, idv_ex.unsqueeze(0))
            return grads

        def moments_single_logit_margin(idv_ex, target):
            label = F.one_hot(target, self.num_classes)

            def logit_margin(p, b, d):
                logits = functional_call(net, (p, b), (d,))
                loss = logits - (logits * label).sum(-1).unsqueeze(-1)
                loss = loss.mean(-1).sum()
                return loss

            grads = grad(logit_margin)(params, buffers, idv_ex.unsqueeze(0))
            return grads

        if self.floss_fn == 'ce':
            moments_fn = moments_single_ce
        elif self.floss_fn == 'nlogits':
            moments_fn = moments_single_nlogits
        elif self.floss_fn == 'back_ce':
            moments_fn = moments_single_back_ce
        elif self.floss_fn == 'logit_margin':
            moments_fn = moments_single_logit_margin

        vmap_moments = vmap(
            moments_fn,
            in_dims=(0, 0),
            randomness='different',
            chunk_size=self.moments_chunk_size,
        )

        # fit second moments in eigenbasis
        total_weight = 0
        for i, batch in enumerate(tqdm(train_loader,
                                       desc="Eigen 2nd Moments: ",
                                       position=0,
                                       leave=True,
                                       total=self.moments_iter)):
            if i == self.moments_iter:
                break

            data = batch['data'].cuda()
            if self.double:
                data = data.double()
            logits = net(data)

            if self.moments_temp == -1:
                probs = F.softmax(torch.ones_like(logits), -1)
            else:
                probs = F.softmax(logits / self.moments_temp, -1).detach()

            raw_probs = F.softmax(logits, -1).detach()

            if self.topk == -1:
                if self.floss_fn == 'back_ce':
                    new_logits = torch.cat((logits, torch.zeros((len(logits), 1)).cuda()), -1)
                    new_probs = F.softmax(new_logits / self.eigen_temp, -1).detach()
                    labels = torch.multinomial(new_probs, 1).squeeze()
                elif self.floss_fn == 'logit_margin':
                    labels = torch.randint(0, self.num_classes, (len(logits),)).cuda()
                else:
                    labels = torch.multinomial(probs, 1).squeeze()

                grads = vmap_moments(data, labels)
                grads = {v: grads[k] for k, v in net.named_parameters()}
                self.optimizer.update_second_moments(grads=grads, labels=labels)
                del grads
                if self.per_class_m1:
                    for kcls in range(self.num_classes):
                        total_weight[kcls] += (labels == kcls).sum()
            else:
                for kcls in range(self.num_classes):
                    labels = torch.ones(len(data)).cuda().long() * kcls
                    grads = vmap_moments(data, labels)
                    grads = {v: grads[k] for k, v in net.named_parameters()}

                    self.optimizer.update_second_moments(grads=grads, ex_weight=probs[:, kcls], labels=labels)
                    del grads
                    if self.per_class_m1:
                        total_weight[kcls] += probs[:, kcls].sum()

            if not self.per_class_m1:
                total_weight += len(data)

        self.optimizer.average_second_moments(total_weight)

        #self.setup_first_eigen_moments(net, val_loader)

    def setup_first_eigen_moments(self, net, loader):
        params = {k: v.detach() for k, v in net.named_parameters()}
        buffers = {k: v.detach() for k, v in net.named_buffers()}

        def moments_single_ce(idv_ex, target):
            label = F.one_hot(target, self.num_classes)
            grads = grad(lambda p, b, d: (-self.moments_temp * F.log_softmax(functional_call(net, (p, b), (d,)) / self.moments_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
            return grads

        vmap_moments = vmap(
            moments_single_ce,
            in_dims=(0, 0),
            randomness='different',
            chunk_size=self.moments_chunk_size,
        )

        total_weight = 0
        for i, batch in enumerate(tqdm(loader,
                                       desc="Eigen 1st Moments: ",
                                       position=0,
                                       leave=True,
                                       total=len(loader))):

            data = batch['data'].cuda()
            if self.double:
                data = data.double()
            labels = batch['label'].cuda()
            grads = vmap_moments(data, labels)
            grads = {v: grads[k] for k, v in net.named_parameters()}
            self.optimizer.update_first_moments(grads=grads, labels=labels)
            total_weight += len(data)
            del grads

        self.optimizer.average_first_moments(total_weight)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        if self.double:
            net.to(torch.double)

        if self.fuse:
            net.fuse_conv_bn_layers()

        net.eval()

        # Use EKFAC
        self.optimizer = EKFAC(
            net,
            eps=self.damping,
            num_classes=self.num_classes,
            sua=self.sua,
            sud=self.sud,
            layer_eps=self.layer_eps,
            eps_type=self.eps_type,
            eigenscale=self.eigenscale,
            featskip=self.featskip,
            layerscale=self.layerscale,
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

            if self.state_path != "None.cpt" and os.path.exists(self.state_path):
                print(f"Loading EKFAC state from {self.state_path}")
                self.optimizer.load_state_dict(torch.load(self.state_path))
            else:
                self.numel = np.sum([p.numel() for p in net.parameters()])

                with torch.enable_grad():

                    self.optimizer.init_hooks(net)

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
                    self.setup_eigen_moments(net, id_loader_dict['train'], id_loader_dict['train'])

                    # clean up again
                    gc.collect()
                    torch.cuda.empty_cache()
                    net.zero_grad()
                    self.optimizer.zero_grad()
                    torch.save(self.optimizer.state_dict(), self.state_path)

            if not os.path.exists(self.spec_path):
                torch.save(self.optimizer.get_spectrum(), self.spec_path)

            if self.reset_m1:
                self.setup_first_eigen_moments(net, id_loader_dict['val'])

            self.optimizer.print_eigens()
            self.optimizer.get_spectrum()

        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):

        data = data.cuda()
        if self.double:
            data = data.double()

        if self.odin:
            with torch.enable_grad():
                data.requires_grad = True
                output = net(data)

                # Calculating the perturbation we need to add, that is,
                # the sign of gradient of cross entropy loss w.r.t. input
                criterion = nn.CrossEntropyLoss()

                labels = output.detach().argmax(axis=1)

                # Using temperature scaling
                output = output / self.odin_temp

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
                new_data = torch.add(data.detach(), gradient, alpha=-self.noise)
                logits = net(new_data)
        else:
            logits = net(data)

        net.zero_grad()
        self.optimizer.zero_grad()

        #ecdf = self.ecdf[None, ...].expand(len(data), -1, -1)

        pred = torch.argmax(logits, -1)

        params = {k: v.detach() for k, v in net.named_parameters()}
        #buffers = {k: v.detach() for k, v in net.named_buffers()}
        buffers = {}

        def grad_single_loss(with_nat=False):
            def grad_single_loss(idv_ex, target):
                label = F.one_hot(target, self.num_classes)

                grads = grad(lambda p, b, d: (-self.grad_temp * F.log_softmax(functional_call(net, (p, b), (d,)) / self.grad_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
                grads = {v: [grads[k].unsqueeze(0), k] for k, v in net.named_parameters()}
                res = self.optimizer.step(grads=grads, inverse=self.inverse, labels=target, return_grads=with_nat, return_feats=False, sandwich=self.sandwich, layers=True)

                return res

            return grad_single_loss
            #self_nak = self.optimizer.dict_dot(grads, nat_grads, return_layers=True)
            #ntk = self.optimizer.dict_dot(grads, grads, return_layers=True)
            #return self_nak, ntk, norm

        def process_new_logits(idv_ex, idv_logits):
            grads = grad(lambda p, b, d: (-F.log_softmax(functional_call(net, (p, b), (d,)) / self.grad_temp, -1) * F.one_hot(idv_logits.argmax(), self.num_classes)).sum())(params, buffers, idv_ex.unsqueeze(0))
            new_params = params.copy()
            for k in grads:
                new_params[k] -= grads[k][0] * 0.00001
            new_logits = functional_call(net, (new_params, buffers), (idv_ex.unsqueeze(0)))
            return new_logits[0]

        def grad_single_uniform(idv_ex):
            grads = grad(lambda p, b, d: -F.log_softmax(functional_call(net, (p, b), (d,)) / self.grad_temp, -1).mean())(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: [grads[k].unsqueeze(0), k] for k, v in net.named_parameters()}
            res = self.optimizer.step(grads=grads, inverse=self.inverse, return_grads=False, return_feats=False, sandwich=self.sandwich, layers=True)
            return res

        def grad_single_back_ce(idv_ex, target):
            label = F.one_hot(target, self.num_classes + 1)

            def back_ce(p, b, d):
                out = functional_call(net, (p, b), (d,)) / self.grad_temp
                out = torch.cat((out, torch.zeros(len(out), 1).cuda()), -1)
                loss = (F.log_softmax(out, -1) * label).sum()
                return loss

            grads = grad(back_ce)(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: [grads[k].unsqueeze(0), k] for k, v in net.named_parameters()}
            res = self.optimizer.step(grads=grads, inverse=self.inverse, labels=target, return_grads=False, return_feats=False, sandwich=self.sandwich, layers=True)
            return res


        def grad_single_logit_margin(idv_ex, target):
            label = F.one_hot(target, self.num_classes)

            def logit_margin(p, b, d):
                logits = functional_call(net, (p, b), (d,))
                loss = logits - (logits * label).sum(-1).unsqueeze(-1)
                loss = loss.mean(-1).sum()
                return loss

            grads = grad(logit_margin)(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: [grads[k].unsqueeze(0), k] for k, v in net.named_parameters()}
            res = self.optimizer.step(grads=grads, inverse=self.inverse, labels=target, return_grads=False, return_feats=False, sandwich=self.sandwich, layers=True)
            return res

        if self.floss_fn == 'ce':
            def process_single_grad(idv_ex, idv_logits):
                res = vmap(grad_single_loss(), in_dims=(None, 0), chunk_size=self.class_chunk_size)(idv_ex, torch.arange(self.num_classes).cuda())
                return res
        elif self.floss_fn == 'back_ce':
            def process_single_grad(idv_ex, idv_logits):
                res = vmap(grad_single_back_ce, in_dims=(None, 0), chunk_size=self.class_chunk_size)(idv_ex, torch.arange(self.num_classes + 1).cuda())
                return res
        elif self.floss_fn == 'logit_margin':
            def process_single_grad(idv_ex, idv_logits):
                res = vmap(grad_single_logit_margin, in_dims=(None, 0), chunk_size=self.class_chunk_size)(idv_ex, torch.arange(self.num_classes).cuda())
                return res


        #new_logits = []
        #for idv_ex, idv_logits in zip(data, logits):
        #    new_logits.append(process_new_logits(idv_ex, idv_logits))

        vmap_grad = vmap(process_single_grad, in_dims=(0, 0,), chunk_size=self.jac_chunk_size, randomness='different')

        #l2_norm, l1_norm, m1_norm, efeats, lfeats, nl_nak = vmap_grad(data, logits) # N x K x L
        l2_norm, l1_norm  = vmap_grad(data, logits) # N x K x L
        #l2r_norm, l1r_norm = vmap(grad_single_uniform, in_dims=(0,), chunk_size=self.jac_chunk_size)(data)
        l2_norm = torch.stack(l2_norm, dim=-1)
        l1_norm = torch.stack(l1_norm, dim=-1)
        #m1_norm = torch.stack(m1_norm, dim=-1)
        #l2r_norm = torch.stack(l2r_norm, dim=-1).sum(-1)

        if self.floss_fn == 'back_ce':
            probs = F.softmax(torch.cat((logits, torch.zeros(len(logits), 1).cuda()), -1) / self.grad_temp, -1)
        else:
            probs = F.softmax(logits / self.grad_temp, -1)

        if self.grad_temp == -1:
            conf = torch.mean(-norm.sum(-1), -1)
        else:
            slayer = self.remove_bot_layers
            tlayer = l2_norm.shape[-1] - self.remove_top_layers

            #conf = ((m1_norm.sum(-1) ** 2 / l2_norm.sum(-1)) * probs).sum(-1)

            #conf = -torch.sum(l2_norm[..., slayer:tlayer].sum(-1) * probs, -1) / self.optimizer.numel
            #conf = -torch.sum(nl_nak * probs, -1)

            if self.floss_fn == 'logit_margin':
                l2_cum = l2_norm.sum(-1).mean(-1)
            else:
                l2_cum = torch.sum(l2_norm.sum(-1) * probs, -1)
            #l1_cum = torch.sum(l1_norm.max(-1)[0] * probs, -1)
            #efeats = torch.sum(efeats * probs[..., None], 1)
            #lfeats = [torch.sum(lf * probs[..., None], 1) for lf in lfeats]# B x L x D
            #m1_cum = torch.sum(m1_norm.sum(-1) * probs, -1)
            #nl_nak = -torch.sum(nl_nak * probs, -1)

            #rao = (m1_norm.sum(-1)**2 / l2_norm.sum(-1))

            conf = -l2_cum
        #conf = -self.lr_model.decision_function(efeats.detach().cpu().numpy())
        #conf = torch.from_numpy(conf)

        rdict = {
            'l2_cum': l2_cum,
            'l2_norm': l2_norm,
            #'l1_cum': l1_cum,
            #'l1_norm': l1_norm,
            'logits': logits,
            #'eigenfeat': efeats,
            #'m1_cum': m1_cum,
            #'m1_norm': m1_norm,
            #'nl_nak': nl_nak,
            #'layerfeat': lfeats,
            #'l2r_norm': l2r_norm,
            'probs': probs,
            #'rao': rao,
        }

        for k in rdict:
            if torch.is_tensor(rdict[k]):
                rdict[k] = rdict[k].cpu().numpy()
            if isinstance(rdict[k], list):
                rdict[k] = [rv.cpu().numpy() for rv in rdict[k]]

        return pred.cpu().numpy(), conf.cpu().numpy(), rdict

        #return pred, conf, {'self_nak': nak, 'conf': conf, 'norm': norm, 'logits': logits, 'ntk': torch.stack(ntk, -1), 'probs': probs}
        #return pred, conf, {'self_nak': nak, 'logits': logits, 'eigenfeat': feat_accum}

