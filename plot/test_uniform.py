import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

slurm_dir = '/h/nng/slurm'
res_dir = 'results/cifar10_resnet18_32x32_test_ood_ood_nak_default/s0/ood/scores'

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='name of dir')
parser.add_argument('-d', '--dataset', help='name of ID dataset')
args = parser.parse_args()

ddir = os.path.join(slurm_dir, args.name, res_dir)
id_data = np.load(os.path.join(ddir, args.dataset + '.npz'), allow_pickle=True)['extra'].item()['self_nak']
for fname in os.listdir(ddir):
    if fname.split('.npz')[0] == args.dataset:
        continue
    print(fname)
    ood_data = np.load(os.path.join(ddir, fname), allow_pickle=True)['extra'].item()['self_nak'].sum(-1).mean(-1)
    labels = np.concatenate((np.zeros(len(id_data)), np.ones(len(ood_data))))
    print(roc_auc_score(labels, np.concatenate((id_data, ood_data))))

