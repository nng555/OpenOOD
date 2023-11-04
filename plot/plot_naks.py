import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

slurm_dir = '/h/nng/slurm'
res_dir = 'results/cifar10_resnet18_32x32_test_ood_ood_nak_default/s0/ood/scores'

parser = argparse.ArgumentParser()
parser.add_argument('-n1', '--name1', help='name of first dir')
parser.add_argument('-n2', '--name2', help='name of second dir')
parser.add_argument('-d', '--dataset', help='name of dataset to plot')
parser.add_argument('-c', '--class_num', help='class number', type=int, default=-1)
args = parser.parse_args()

d1 = np.load(os.path.join(slurm_dir, args.name1, res_dir, args.dataset + '.npz'), allow_pickle=True)
d2 = np.load(os.path.join(slurm_dir, args.name2, res_dir, args.dataset + '.npz'), allow_pickle=True)

if args.class_num == -1:
    plt.scatter(-d1['conf'], -d2['conf'], s=1, alpha=0.2)
else:
    plt.scatter(d1['extra'].item()['self_nak'].sum(-1)[:, args.class_num], d2['extra'].item()['self_nak'].sum(-1)[:, args.class_num], s=1, alpha=0.2)
plt.xscale('log')
plt.yscale('log')
plt.savefig('out.png')
