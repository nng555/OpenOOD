import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from colour import Color
from scipy.special import softmax
import seaborn as sns
from sklearn.metrics import roc_auc_score
from pathlib import Path

slurm_dir = '/h/nng/slurm'
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='name of dir')
parser.add_argument('-d', '--dataset', help='name of ID dataset')
parser.add_argument('-t', '--temp', type=float, help='temp')
args = parser.parse_args()

if args.dataset == 'cifar10':
    if 'nin' in args.name:
        res_dir = 'results/cifar10_nin_test_ood_ood_nak_default/s0/ood/scores'
    elif 'vgg' in args.name:
        res_dir = 'results/cifar10_vgg_test_ood_ood_nak_default/s0/ood/scores'
    else:
        res_dir = 'results/cifar10_resnet18_32x32_test_ood_ood_nak_default/s0/ood/scores'
elif args.dataset == 'mnist':
    res_dir = 'results/mnist_lenet_test_ood_ood_nak_default/s0/ood/scores'
elif args.dataset == 'cifar100':
    if '50' in args.name:
        res_dir = 'results/cifar100_resnet50_test_ood_ood_nak_default/s0/ood/scores'
    else:
        res_dir = 'results/cifar100_resnet18_32x32_test_ood_ood_nak_default/s0/ood/scores'

ddir = os.path.join(slurm_dir, args.name, res_dir)
id_data = np.load(os.path.join(ddir, args.dataset + '.npz'), allow_pickle=True)['extra'].item()
id_norm = (id_data['m2_norm'] * softmax(id_data['logits'] / args.temp, -1)[:, :, None]).sum(1)
Path(args.name.split('/')[-1]).mkdir(parents=True, exist_ok=True)

for fname in os.listdir(ddir):
    print(fname)
    if fname == args.dataset + '.npz':
        continue
    ood_data = np.load(os.path.join(ddir, fname), allow_pickle=True)['extra'].item()
    ood_norm = (ood_data['m2_norm'] * softmax(ood_data['logits'] / args.temp, -1)[:, :, None]).sum(1)
    labels = np.concatenate((np.zeros(len(id_norm)), np.ones(len(ood_norm))))

    for i in range(id_norm.shape[-1]):
        print(roc_auc_score(labels, np.concatenate((id_norm[:, i], ood_norm[:, i]))))

    print()
