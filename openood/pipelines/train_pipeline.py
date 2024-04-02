import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import openood.utils.comm as comm
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger

from openood.postprocessors import EKFAC

class TrainPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set random seed
        print(f"Seeding with {self.config.seed}")
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        #torch.backends.cudnn.benchmark = False
        #torch.use_deterministic_algorithms(True)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        train_loader, val_loader = loader_dict['train'], loader_dict['val']

        test_loader = loader_dict['test']
        if self.config.dataset.comp_index != -1:
            if self.config.dataset.ood_comp != self.config.dataset.name:
                ood_loader_dict = get_ood_dataloader(self.config)
                if self.config.dataset.ood_comp in ood_loader_dict['nearood']:
                    comp_img = ood_loader_dict['nearood'][self.config.dataset.ood_comp].dataset.imglist[self.config.dataset.comp_index]
                elif self.config.dataset.ood_comp in ood_loader_dict['farood']:
                    comp_img = ood_loader_dict['farood'][self.config.dataset.ood_comp].dataset.imglist[self.config.dataset.comp_index]
                else:
                    raise Exception(f"Dataset {self.config.dataset.ood_comp} not found")
                del ood_loader_dict
            else:
                comp_img = test_loader.dataset.imglist[self.config.dataset.comp_index]
            orig_label = int(comp_img.strip('\n').split(' ')[-1])
            comp_img = comp_img.split(' ')[0] + ' ' + str(self.config.dataset.comp_label) + 'extra\n'
            print(f"Adding img index {self.config.dataset.comp_index} and replacing label {orig_label} with regret label {self.config.dataset.comp_label}: {comp_img}")
            train_loader.dataset.imglist.append(comp_img)

        # init network
        net = get_network(self.config.network)
        if self.config.pbrf:
            print("Building extra model for pbrf labels")
            base_net = get_network(self.config.network)
            # freeze both batchnorm layers
            #for mod in net.modules():
            #    if isinstance(mod, nn.BatchNorm2d):
            #        mod.track_running_stats=False
        else:
            base_net = None

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, val_loader, self.config, base_net)
        evaluator = get_evaluator(self.config)

        if comm.is_main_process():
            # init recorder
            recorder = get_recorder(self.config)

            print('Start training...', flush=True)

        if self.config.dataset.comp_index != -1:
            net.eval()
            comp_data = train_loader.dataset[len(train_loader.dataset) - 1]
            comp_out = net(comp_data['data'].unsqueeze(0).cuda())
            label = comp_data['label']
            orig_comp_log_prob = F.log_softmax(comp_out[0])[label]
            orig_comp_prob = F.softmax(comp_out[0])[label]
            comp_logit = comp_out[0][label]
            comp_prob = F.softmax(comp_out[0])
            print(f"Comp logit: {comp_logit}\nFull logits: {comp_out[0]}")
            print(f"Comp prob: {comp_prob[label]}\nFull probs: {comp_prob}")
            print(f"Regret: 0")

            net.train()

        new_nepochs = int(self.config.optimizer.num_epochs / (1 - self.config.dataset.prune))
        if self.config.dataset.prune != 0.0:
            print(f"Increasing num_epochs from {self.config.optimizer.num_epochs} to {new_nepochs}")

        swa_params = None
        swa_var = None

        for epoch_idx in range(1, new_nepochs + 1):
            # train and eval the model
            if self.config.trainer.name == 'mos':
                net, train_metrics, num_groups, group_slices = \
                    trainer.train_epoch(epoch_idx)
                val_metrics = evaluator.eval_acc(net,
                                                 val_loader,
                                                 train_loader,
                                                 epoch_idx,
                                                 num_groups=num_groups,
                                                 group_slices=group_slices)
            elif self.config.trainer.name in ['cider', 'npos']:
                net, train_metrics = trainer.train_epoch(epoch_idx)
                # cider and npos only trains the backbone
                # cannot evaluate ID acc without training the fc layer
                val_metrics = train_metrics
            else:
                net, train_metrics = trainer.train_epoch(epoch_idx)
                net.eval()
                val_metrics = evaluator.eval_acc(net, val_loader, None,
                                                 epoch_idx)
                net.train()

            comm.synchronize()
            if comm.is_main_process():
                # save model and report the result
                if self.config.recorder.save_any_model:
                    recorder.save_model(net, val_metrics, epoch_idx)
                recorder.report(train_metrics, val_metrics)

            if self.config.dataset.comp_index != -1:
                net.eval()
                comp_data = train_loader.dataset[len(train_loader.dataset) - 1]
                comp_out = net(comp_data['data'].unsqueeze(0).cuda())
                label = comp_data['label']
                comp_log_prob = F.log_softmax(comp_out[0])[label]
                comp_logit = comp_out[0][label]
                comp_prob = F.softmax(comp_out[0])
                print(f"Comp logit: {comp_logit} \t Full logits: {comp_out[0]}")
                print(f"Comp prob: {comp_prob[label]} \t Full probs: {comp_prob}")
                print(f"Regret: {comp_log_prob - orig_comp_log_prob}")
                print(f"Prob diff: {comp_prob[label] - orig_comp_prob}")
                net.train()

            if self.config.trainer.swag != -1 and epoch_idx >= self.config.trainer.swag:
                nswa = epoch_idx - self.config.trainer.swag
                print(f"Adding SWAG model {nswa + 1}")
                new_swa_params = net.state_dict()
                new_swa_var = net.state_dict()
                for k in new_swa_var:
                    new_swa_var[k] = new_swa_var[k]**2
                if swa_params is None:
                    swa_params = new_swa_params
                    swa_var = new_swa_var
                else:
                    for k in swa_params:
                        swa_params[k] = swa_params[k] * (nswa / (nswa + 1)) + new_swa_params[k] / (nswa + 1)
                        swa_var[k] = swa_var[k] * (nswa / (nswa + 1)) + new_swa_var[k] / (nswa + 1)

        if comm.is_main_process():
            recorder.summary()
            print(u'\u2500' * 70, flush=True)

            # evaluate on test set
            print('Start testing...', flush=True)

        net.eval()
        test_metrics = evaluator.eval_acc(net, test_loader)

        if comm.is_main_process():

            print('\nComplete Evaluation, Last accuracy {:.2f}'.format(
                100.0 * test_metrics['acc']),
                  flush=True)
            print('Completed!', flush=True)

            if self.config.trainer.swag != -1:
                torch.save(swa_params, os.path.join(recorder.output_dir, "swa.ckpt"))
                for k in swa_var:
                    swa_var[k] = swa_var[k] - swa_params[k]**2
                torch.save(swa_var, os.path.join(recorder.output_dir, "swa_var.ckpt"))
