import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns
from sklearn.metrics import roc_auc_score
from pathlib import Path

slurm_dir = '/h/nng/slurm'
res_dir = 'results/cifar10_resnet18_32x32_test_ood_ood_nak_default/s0/ood/scores'
#res_dir = 'results/mnist_lenet_test_ood_ood_nak_default/s0/ood/scores'

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='name of dir')
parser.add_argument('-d', '--dataset', help='name of ID dataset')
args = parser.parse_args()

ddir = os.path.join(slurm_dir, args.name, res_dir)
id_data = np.load(os.path.join(ddir, args.dataset + '.npz'), allow_pickle=True)['extra'].item()['eigenfeat']
Path(args.name.split('/')[-1]).mkdir(parents=True, exist_ok=True)

for fname in os.listdir(ddir):
    print(fname)
    if fname == args.dataset + '.npz':
        continue
    ood_data = np.load(os.path.join(ddir, fname), allow_pickle=True)['extra'].item()['eigenfeat']
    labels = np.concatenate((np.zeros(len(id_data)), np.ones(len(ood_data))))
    aurocs = []
    for i in range(id_data.shape[1]):
        aurocs.append(roc_auc_score(labels, np.concatenate((id_data[:, i], ood_data[:, i]))))
    plt.clf()
    plt.cla()
    plt.plot(np.arange(len(aurocs)), aurocs)
    plt.savefig(args.name.split('/')[-1] + '/' + fname + '.png')

