import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing, constant_divide


class BaseTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
            config: Config, base_net=None) -> None:

        self.net = net
        self.base_net = base_net
        self.pbrf = config.pbrf
        assert not (self.pbrf and base_net is None)
        if self.pbrf:
            base_net.eval()
            self.numel = sum([p.numel() for p in base_net.parameters()])
        self.train_loader = train_loader
        self.damping = 5e-4
        self.config = config
        self.num_classes = self.config.dataset.num_classes

        if config.optimizer.name == 'sgd':
            self.optimizer = torch.optim.SGD(
                net.parameters(),
                config.optimizer.lr,
                momentum=config.optimizer.momentum,
                weight_decay=config.optimizer.weight_decay,
                nesterov=True,
            )
        elif config.optimizer.name == 'adam':
            self.optimizer = torch.optim.Adam(
                net.parameters(),
                config.optimizer.lr,
                weight_decay=config.optimizer.weight_decay,
            )

        """
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )
        """

        if config.optimizer.schedule == 'stepwise':
            full_loader_size = int(len(train_loader) / (1 - self.config.dataset.prune))
            if self.config.dataset.prune != 0.0:
                print(f"Increasing loader size from {len(train_loader)} to {full_loader_size}")

            if self.config.pbrf:
                div_steps = [dstep * full_loader_size for dstep in [25,50,75]]
            else:
                div_steps = [dstep * full_loader_size for dstep in [60,120,180,240]]

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: constant_divide(
                    step,
                    div_steps,
                    [5,5,5,5],
                    warmup=600,
                )
            )

        elif config.optimizer.schedule == 'swag':
            div_steps = [dstep * int(len(train_loader)) for dstep in [60,120,160]]
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: constant_divide(
                    step,
                    div_steps,
                    [5,5,0.4],
                    warmup=300,
                )
            )
        else:
            self.scheduler = None




    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            if self.pbrf:
                pbrf_targets = F.log_softmax(self.base_net(data).detach(), -1)
                pbrf = batch['pbrf']
                pbrf_index = None
                for i, _pbrf in enumerate(pbrf):
                    if _pbrf:
                        pbrf_index = i
                        pbrf_targets[i] = F.one_hot(target[i], self.num_classes)

            # forward
            logits_classifier = self.net(data)
            if self.pbrf:
                damping_loss = 0
                log_prob_classifier = F.log_softmax(logits_classifier, -1)
                for base_p, net_p in zip(self.base_net.parameters(), self.net.parameters()):
                    damping_loss += torch.sum((base_p.detach() - net_p)**2)
                if pbrf_index is not None:
                    loss = F.kl_div(log_prob_classifier[:pbrf_index], pbrf_targets[:pbrf_index], reduction='sum', log_target=True) / self.num_classes
                    loss += F.cross_entropy(logits_classifier[pbrf_index], target[pbrf_index])
                    loss += F.kl_div(log_prob_classifier[pbrf_index + 1:], pbrf_targets[pbrf_index + 1:], reduction='sum', log_target=True) / self.num_classes
                    loss /= len(data)
                else:
                    loss = F.kl_div(log_prob_classifier, pbrf_targets, reduction='batchmean', log_target=True)
                loss += self.damping * damping_loss * len(data) / len(train_dataiter)
            else:
                loss = F.cross_entropy(logits_classifier, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)
        if self.scheduler is not None:
            print(self.scheduler.get_last_lr())

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
