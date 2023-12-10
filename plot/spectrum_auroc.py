import argparse
import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns
from sklearn.metrics import roc_auc_score
from pathlib import Path
import torch

slurm_dir = '/h/nng/slurm'
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='name of dir')
parser.add_argument('-d', '--dataset', help='name of ID dataset')
parser.add_argument('-s', '--spec', help='name of spectrum')
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
id_data = np.load(os.path.join(ddir, args.dataset + '.npz'), allow_pickle=True)['extra'].item()['eigenfeat']
Path(args.name.split('/')[-1]).mkdir(parents=True, exist_ok=True)

lr_model = pkl.load(open('/h/nng/projects/OpenOOD/results/ekfac/' + args.spec + '_lr.pkl', 'rb'))
spec = torch.load('/h/nng/projects/OpenOOD/results/ekfac/' + args.spec + '_spec.cpt', map_location='cpu')
spec = torch.cat(spec).numpy()
spec += 1e-30
eigen_min = spec[spec != 0].min()
eigen_max = spec.max()

threshes = np.logspace(np.log10(eigen_min), np.log10(eigen_max), num=100)
idxs = []
for i in range(len(threshes) - 1):
    if ((threshes[i] < spec) & (threshes[i + 1] > spec)).sum() > 0:
        idxs.append(threshes[i])

plt.plot(np.arange(len(lr_model.coef_[0])), lr_model.coef_[0])
plt.savefig(args.name.split('/')[-1] + '/weights.png')
plt.clf()
plt.cla()

fig1, ax1 = plt.subplots()

for fname in os.listdir(ddir):
    fig2, ax2 = plt.subplots()
    print(fname)
    if fname == args.dataset + '.npz':
        continue
    ood_data = np.load(os.path.join(ddir, fname), allow_pickle=True)['extra'].item()['eigenfeat']
    labels = np.concatenate((np.zeros(len(id_data)), np.ones(len(ood_data))))
    aurocs = []
    xs = []
    for i in range(len(idxs)):
        auroc = roc_auc_score(labels, np.concatenate((id_data[:, i], ood_data[:, i])))
        if auroc < 0.51 and auroc > 0.49:
            continue
        aurocs.append(auroc)
        xs.append(idxs[i])
    ax2.plot(xs, aurocs)
    ax1.plot(xs, aurocs, label=fname, alpha=0.7)
    ax2.set_xscale('log')
    fig2.savefig(args.name.split('/')[-1] + '/' + fname + '.png')
ax1.set_xscale('log')
fig1.legend()
fig1.savefig(args.name.split('/')[-1] + '/full.png')

