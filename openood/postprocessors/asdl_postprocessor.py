import os
from tqdm import tqdm
from typing import Any
import yaml

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as T
import numpy as np
from torch.func import functional_call, vmap, grad, jacrev

from asdl.precondition import PreconditioningConfig, NewtonGradientMaker
from laplace import Laplace

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict

class ASDLPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ASDLPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.activation_log = None
        self.setup_flag = False
        self.damping = self.args.damping
        self.topk = self.args.topk
        assert self.num_classes % self.topk == 0

        # multiply loss to get larger scaled eigenvalues
        self.loss_scale = self.args.loss_scale

        self.temp = self.args.temp

        self.loss_name = self.args.loss_name
        if self.loss_name == 'ce':
            self.loss_fn = self.ce_loss
        elif self.loss_name == 'logit_margin':
            assert self.temp == 1, "No temperature scaling allowed for logit losses"
            self.loss_fn = self.logit_margin_loss
        else:
            raise NotImplementedError

        self.state_name = self.args.state_name
        self.root_dir = self.args.root_dir
        self.fuse = self.args.fuse

    def ce_loss(self, logits, targets=None):
        logits = logits / self.temp

        if targets is None:
            if self.topk == -1:
                probs = F.softmax(logits, dim=-1).detach()
                targets = torch.multinomial(probs, 1).squeeze()
            else:
                logits = torch.topk(logits, self.topk, -1)[0]
                probs = F.softmax(logits, dim=-1).detach()
                targets = torch.multinomial(probs, 1).squeeze()
        elif self.topk != -1:
            logits = torch.topk(logits, self.topk, -1)[0]

        loss = F.cross_entropy(logits, targets, reduction="sum")
        return self.loss_scale * loss

    def logit_margin_loss(self, logits, targets=None):
        if targets is None:
            if self.topk == -1:
                #probs = F.softmax(logits, dim=-1).detach()
                #targets = torch.multinomial(probs, 1).squeeze()
                targets = torch.randint(0, self.num_classes, (len(logits),)).cuda()
            else:
                logits = torch.sort(logits, -1)[0]
                logits = logits[:, ::int(self.num_classes/self.topk)]
                targets = torch.randint(0, len(logits[0]), (len(logits),)).cuda()
        elif self.topk != -1:
            logits = torch.sort(logits, -1)[0]
            logits = logits[:, ::int(self.num_classes/self.topk)]

        # (z - zi).mean()
        loss = logits - logits.take_along_dim(targets.unsqueeze(-1), -1)
        loss = loss.mean(-1).sum()
        return self.loss_scale * loss

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):

        # maybe need to modify this depending on whether we use the full dataset
        self.ntrain = len(id_loader_dict['train'].sampler)

        if self.fuse:
            net.fuse_conv_bn_layers()

        net.eval()

        self.la = Laplace(net, 'classification', subset_of_weights='all')
        self.la.fit(id_loader_dict['train'])
        import ipdb; ipdb.set_trace()

        config = PreconditioningConfig(data_size=id_loader_dict['train'].batch_size, damping=self.damping)
        self.gm = NewtonGradientMaker(net, config)

        if not self.setup_flag:

            if os.path.exists(os.path.join(self.root_dir, self.state_name, 'state')):
                self.analog.initialize_from_log()
            else:
                for i, batch in enumerate(tqdm(id_loader_dict['train'],
                                               desc='Hessian: ',
                                               position=0,
                                               leave=True,
                                               total=len(id_loader_dict['train']))):
                    data = batch['data'].cuda()
                    targets = batch['label'].cuda()

                    dummy_out = self.gm.setup_model_call(net, data)
                    self.gm.setup_loss_call(self.loss_fn, dummy_out, targets)
                    out, loss = self.gm.forward_and_backward()

                    import ipdb; ipdb.set_trace()

        else:
            pass

    def build_grad_fn(self, net, params, buffers):
        def grad_single(idv_ex, target):
            grads = grad(
                lambda p, b, d: self.loss_fn(
                    functional_call(net, (p, b), (d,)),
                    target.unsqueeze(-1),
                )
            )(params, buffers, idv_ex.unsqueeze(0))
            return grads

        vmap_grad = vmap(grad_single, in_dims=(0, 0,), chunk_size=8)

        return vmap_grad

    def postprocess(self, net: nn.Module, data: Any):

        pcount = sum(p.numel() for p in net.parameters())
        data = data.cuda()

        logits = net(data).detach()
        probs = F.softmax(logits / self.temp, -1)
        pred = torch.argmax(logits, -1)

        if_scores = []

        nclasses = self.num_classes if self.topk == -1 else self.topk

        for k in range(nclasses):
            with self.analog(data_id=self.id_gen(data)):
                logits = net(data)
                targets = torch.ones(len(data)).cuda().long() * k
                net.zero_grad()
                loss = self.loss_fn(logits, targets)
                loss.backward()

            test_log = self.analog.get_log()
            if_scores.append(
                self.analog.influence.compute_self_influence_norm(
                    test_log,
                    damping=self.damping,
                )
            )

        if_scores = torch.stack(if_scores, dim=-1)

        if self.loss_name == 'ce':
            # measure change in probability space
            conf = -(if_scores * probs).sum(-1)
        elif self.loss_name == 'logit_margin':
            # measure change in logit space
            conf = -if_scores.mean(-1)
        else:
            raise NotImplementedError

        extras = {
            'l2_norm': if_scores.cpu().numpy(),
            'logits': logits.detach().cpu().numpy(),
            'probs': probs.cpu().numpy(),
        }

        return pred.cpu().numpy(), conf.cpu().numpy(), extras

