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
    #if '50' in args.name:
    #    res_dir = 'results/cifar100_resnet50_test_ood_ood_nak_default/s0/ood/scores'
    #else:
    res_dir = 'results/cifar100_resnet18_32x32_test_ood_ood_nak_default/s0/ood/scores'

ddir = os.path.join(slurm_dir, args.name, res_dir)
id_data = np.load(os.path.join(ddir, args.dataset + '.npz'), allow_pickle=True)['extra'].item()
id_data = id_data['layerfeat']
Path(args.name.split('/')[-1]).mkdir(parents=True, exist_ok=True)

for fname in os.listdir(ddir):
    print(fname)
    if fname == args.dataset + '.npz':
        continue
    ood_data = np.load(os.path.join(ddir, fname), allow_pickle=True)['extra'].item()['layerfeat']
    labels = np.concatenate((np.zeros(len(id_data)), np.ones(len(ood_data))))

    blue = Color('blue')
    colors = list(blue.range_to(Color('green'), id_data.shape[1]))

    taurocs = []
    aurocs = []
    idx = 0
    idxs = []

    plt.clf()
    plt.cla()
    for lnum, c in zip(range(id_data.shape[1]), colors):
        for i in range(id_data.shape[2]):
            aurocs.append(roc_auc_score(labels, np.concatenate((id_data[:, lnum, i], ood_data[:, lnum, i]))))
            idxs.append(idx)
            idx += 1
        taurocs.append(roc_auc_score(labels, np.concatenate((id_data[:, lnum, :].sum(-1), ood_data[:, lnum, :].sum(-1)))))
        plt.plot(idxs, aurocs, color=str(c))
        idx += 25
        aurocs = []
        idxs = []

    plt.plot([i * (id_data.shape[2] + 25) + id_data.shape[2] // 2 for i in range(id_data.shape[1])], taurocs, color='black')
    print(max(taurocs))

    plt.savefig(args.name.split('/')[-1] + '/' + fname + '.png')

