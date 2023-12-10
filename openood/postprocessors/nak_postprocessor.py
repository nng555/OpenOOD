from typing import Any
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
        self.center = self.args.center
        self.state_path = self.args.state_path + '.cpt'
        self.grad_path = self.args.state_path + '_grad.cpt'
        self.spec_path = self.args.state_path + '_spec.cpt'
        self.ecdf_path = self.args.state_path + '_ecdf.cpt'
        self.lr_model_path = self.args.state_path + '_lr.pkl'
        self.jac_chunk_size = self.args.jac_chunk_size
        self.empirical = self.args.empirical
        self.moments_chunk_size = self.args.moments_chunk_size
        self.class_chunk_size = self.args.class_chunk_size

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
                loss = self.floss_temp * F.cross_entropy(logits / self.floss_temp, labels, reduction='sum') / self.ntrain
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

                loss = self.eigen_temp * F.cross_entropy(logits / self.eigen_temp, labels, reduction='sum')
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
                    loss = self.eigen_temp * F.cross_entropy(logits / self.eigen_temp, labels, reduction='sum')
                    loss.backward(retain_graph=(kcls != self.num_classes - 1))

                    # centering happens inside second moment estimator
                    # weight by probabilities
                    self.optimizer.update_state(ex_weight=probs[:, kcls], labels=labels)
                    if self.per_class_m1:
                        total_weight[kcls] += probs[:, kcls].sum()

            if not self.per_class_m1:
                total_weight += len(logits)

        self.optimizer.average_state(total_weight)

    def setup_eigen_moments(self, net, train_loader, val_loader, moment=2):
        params = {k: v.detach() for k, v in net.named_parameters()}
        buffers = {k: v.detach() for k, v in net.named_buffers()}

        def moments_single_ce(idv_ex, target):
            label = F.one_hot(target, self.num_classes)
            grads = grad(lambda p, b, d: (-self.moments_temp * F.log_softmax(functional_call(net, (p, b), (d,)) / self.moments_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
            return grads

        def moments_single_logits(idv_ex, target):
            # use a multiplicative mask inside vmap
            label = F.one_hot(target, self.num_classes)
            def logits_loss(params, buffers, data):
                out = functional_call(net, (params, buffers), (data,)) / self.moments_temp
                loss = 0.5 * (out**2 * label).sum() + (out * label).sum() * out.sum()
                return loss
            grads = grad(lambda p, b, d: logits_loss(p, b, d))(params, buffers, idv_ex.unsqueeze(0))
            return grads

        if self.floss_fn == 'ce':
            moment_fn = moments_single_ce
        elif self.floss_fn == 'logits':
            moment_fn = moments_single_logits

        vmap_moments = vmap(
            moment_fn,
            in_dims=(0, 0),
            randomness='different',
            chunk_size=self.moments_chunk_size,
        )

        # fit second moments in eigenbasis
        total_weight = 0
        for i, batch in enumerate(tqdm(loader,
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
                    # center gradients before projection
                    if self.center:
                        grads = {v: grads[k] - self.avg_grad[k][kcls] for k, v in net.named_parameters()}
                    else:
                        grads = {v: grads[k] for k, v in net.named_parameters()}

                    self.optimizer.update_second_moments(grads=grads, ex_weight=probs[:, kcls], labels=labels)
                    del grads
                    if self.per_class_m1:
                        total_weight[kcls] += probs[:, kcls].sum()

            if not self.per_class_m1:
                total_weight += len(data)

        self.optimizer.average_second_moments(total_weight)

        total_weight = 0
        for i, batch in enumerate(tqdm(val_loader,
                                       desc="Eigen 1st Moments: ",
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
                labels = torch.multinomial(probs, 1).squeeze()
                grads = vmap_moments(data, labels)
                grads = {v: grads[k] for k, v in net.named_parameters()}
                self.optimizer.update_first_moments(grads=grads, labels=labels)
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

                    self.optimizer.update_first_moments(grads=grads, ex_weight=probs[:, kcls], labels=labels)
                    del grads
                    if self.per_class_m1:
                        total_weight[kcls] += probs[:, kcls].sum()

            if not self.per_class_m1:
                total_weight += len(data)

        self.optimizer.average_first_moments(total_weight)

    def setup_weights(self, net, val_id_loader, val_ood_loader):

        params = {k: v.detach() for k, v in net.named_parameters()}
        buffers = {k: v.detach() for k, v in net.named_buffers()}

        def grad_single_loss(idv_ex, target):
            label = F.one_hot(target, self.num_classes)

            # calculate and center gradients
            grads = grad(lambda p, b, d: (-F.log_softmax(functional_call(net, (p, b), (d,)) / self.grad_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
            grads = {v: grads[k].unsqueeze(0) for k, v in net.named_parameters()}

            _, _, _, _, efeat, _ = self.optimizer.step(grads=grads, inverse=self.inverse, l2=True, labels=target, return_feats=True, sandwich=self.sandwich, layers=True)
            return efeat

        def process_single_grad(idv_ex):
            res = vmap(grad_single_loss, in_dims=(None, 0), chunk_size=self.class_chunk_size)(idv_ex, torch.arange(self.num_classes).cuda())
            return res

        vmap_grad = vmap(process_single_grad, in_dims=(0,), chunk_size=self.jac_chunk_size, randomness='different')


        def process_loader(loader, name):
            feats = []
            for i, batch in enumerate(tqdm(loader,
                                           desc=name + " Eigenfeatures: ",
                                           position=0,
                                           leave=True,
                                           total=len(loader))):

                data = batch['data'].cuda()
                logits = net(data)
                probs = F.softmax(logits / self.grad_temp, -1)

                eigenfeat = vmap_grad(data)
                eigenfeat = torch.sum(eigenfeat * probs[..., None], 1)
                feats.append(eigenfeat)

            feats = torch.cat(feats)
            return feats

        id_feats = process_loader(val_id_loader, "ID").detach().cpu().numpy()
        ood_feats = process_loader(val_ood_loader, "OOD").detach().cpu().numpy()
        X = np.concatenate((id_feats, ood_feats))
        Y = np.concatenate((np.zeros(len(id_feats)), np.ones(len(ood_feats))))

        #scaler = RobustScaler().fit(X)
        #X = scaler.transform(X)
        model = LogisticRegression(C=1, maxiter=10000)
        model.fit(X, Y)

        return model

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        if self.double:
            net.to(torch.double)

        if self.fuse:
            net.fuse_conv_bn_layers()

        net.eval()
        print(net)

        # Use EKFAC
        if not self.full_fim:
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

        # use NNGeometry Full Implicit FIM
        else:
            from nngeometry.metrics import FIM_MonteCarlo
            from nngeometry.object import PMatImplicit
            from nngeometry.layercollection import LayerCollection
            from nngeometry.object.vector import random_pvector

            self.fim = FIM_MonteCarlo(model=net,
                                      loader=id_loader_dict['train'],
                                      representation=PMatImplicit,
                                      device='cuda')

            layer_collection = LayerCollection.from_model(net)
            v = random_pvector(LayerCollection.from_model(net), device='cuda')
            import time
            start = time.time()
            out = self.fim.mv(v)
            end = time.time()
            print(end - start)

            import ipdb; ipdb.set_trace()


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

            # accumulate means for centering
            if self.whiten:
                self.optimizer.init_hooks(net)
                res = self.setup_means(net, id_loader_dict['train'])
                net._whiten(res)

                # rebuild optimizer
                self.optimizer.clear_hooks()
                self.optimizer.clear_cache()
                self.optimizer.zero_grad()
                self.optimizer.rebuild(net)

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

            """
            if os.path.exists(self.lr_model_path):
                self.lr_model = pkl.load(open(self.lr_model_path, 'rb'))
            else:
                self.lr_model = self.setup_weights(net, id_loader_dict['val'], ood_loader_dict['val'])
                pkl.dump(self.lr_model, open(self.lr_model_path, 'wb'))
            """

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

                # calculate and center gradients
                grads = grad(lambda p, b, d: (-F.log_softmax(functional_call(net, (p, b), (d,)) / self.grad_temp, -1) * label).sum())(params, buffers, idv_ex.unsqueeze(0))
                grads = {v: [grads[k].unsqueeze(0), k] for k, v in net.named_parameters()}

                nat_grads, l2_norm, l1_norm, m1_norm, efeat, lfeat = self.optimizer.step(grads=grads, inverse=self.inverse, l2=True, labels=target, return_feats=True, sandwich=self.sandwich, layers=True)
                if with_nat:
                    return {k: v[0] for k, v in nat_grads.items()}
                else:
                    return l2_norm, l1_norm, m1_norm, efeat, lfeat

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

        def process_single_grad(idv_ex, idv_logits):
            res = vmap(grad_single_loss(), in_dims=(None, 0), chunk_size=self.class_chunk_size)(idv_ex, torch.arange(self.num_classes).cuda())
            return res

        #new_logits = []
        #for idv_ex, idv_logits in zip(data, logits):
        #    new_logits.append(process_new_logits(idv_ex, idv_logits))

        vmap_grad = vmap(process_single_grad, in_dims=(0, 0,), chunk_size=self.jac_chunk_size, randomness='different')

        l2_norm, l1_norm, m1_norm, efeats, lfeats = vmap_grad(data, logits) # N x K x L
        l2_norm = torch.stack(l2_norm, dim=-1)
        l1_norm = torch.stack(l1_norm, dim=-1)
        m1_norm = torch.stack(m1_norm, dim=-1)

        probs = F.softmax(torch.stack(logits) / self.grad_temp, -1)

        if self.grad_temp == -1:
            conf = torch.mean(-norm.sum(-1), -1)
        else:
            slayer = self.remove_bot_layers
            tlayer = l2_norm.shape[-1] - self.remove_top_layers
            conf = torch.sum(l2_norm[..., slayer:tlayer].sum(-1) * probs, -1)

            l2_cum = torch.sum(l2_norm.sum(-1) * probs, -1)
            l1_cum = torch.sum(l1_norm.max(-1)[0] * probs, -1)
            efeats = torch.sum(efeats * probs[..., None], 1)
            lfeats = [torch.sum(lf * probs[..., None], 1) for lf in lfeats]# B x L x D
            m1_cum = torch.sum(m1_norm.sum(-1) * probs, -1)

        #conf = -self.lr_model.decision_function(efeats.detach().cpu().numpy())
        #conf = torch.from_numpy(conf)

        rdict = {
            'l2_cum': l2_cum,
            'l2_norm': l2_norm,
            'l1_cum': l1_cum,
            'l1_norm': l1_norm,
            'logits': logits,
            'eigenfeat': efeats,
            'm1_cum': m1_cum,
            'm1_norm': m1_norm,
            #'layerfeat': lfeats,
        }

        for k in rdict:
            if torch.is_tensor(rdict[k]):
                rdict[k] = rdict[k].cpu().numpy()
            if isinstance(rdict[k], list):
                rdict[k] = [rv.cpu().numpy() for rv in rdict[k]]

        return pred.cpu().numpy(), conf.cpu().numpy(), rdict

        #return pred, conf, {'self_nak': nak, 'conf': conf, 'norm': norm, 'logits': logits, 'ntk': torch.stack(ntk, -1), 'probs': probs}
        #return pred, conf, {'self_nak': nak, 'logits': logits, 'eigenfeat': feat_accum}

