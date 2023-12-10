import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

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
    if '50' in args.name:
        res_dir = 'results/cifar100_resnet50_test_ood_ood_nak_default/s0/ood/scores'
    else:
        res_dir = 'results/cifar100_resnet18_32x32_test_ood_ood_nak_default/s0/ood/scores'

ddir = os.path.join(slurm_dir, args.name, res_dir)
id_data = np.load(os.path.join(ddir, args.dataset + '.npz'), allow_pickle=True)['extra'].item()
id_norm = id_data['m2_norm'].sum(-1)
for fname in os.listdir(ddir):
    if fname.split('.npz')[0] == args.dataset:
        continue
    print(fname)
    ood_data = np.load(os.path.join(ddir, fname), allow_pickle=True)['extra'].item()
    ood_norm = ood_data['m2_norm'].sum(-1)
    import ipdb; ipdb.set_trace()
    labels = np.concatenate((np.zeros(len(id_norm)), np.ones(len(ood_norm))))
    print(roc_auc_score(labels, np.concatenate((id_norm.mean(-1), ood_norm.mean(-1)))))
