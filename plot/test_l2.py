import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns
from sklearn.metrics import roc_auc_score

slurm_dir = '/h/nng/slurm'
res_dir = 'results/cifar10_resnet18_32x32_test_ood_ood_nak_default/s0/ood/scores'
#res_dir = 'results/mnist_lenet_test_ood_ood_nak_default/s0/ood/scores'

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='name of dir')
parser.add_argument('-d', '--dataset', help='name of ID dataset')
args = parser.parse_args()

ddir = os.path.join(slurm_dir, args.name, res_dir)
id_data = np.load(os.path.join(ddir, args.dataset + '.npz'), allow_pickle=True)['extra'].item()
id_nak = np.sum(id_data['self_nak'].sum(-1) * id_data['probs'], -1)
id_norm = id_data['conf']

for fname in os.listdir(ddir):
    if fname == args.dataset + '.npz':
        continue
    ood_data = np.load(os.path.join(ddir, fname), allow_pickle=True)['extra'].item()
    ood_nak = np.sum(ood_data['self_nak'].sum(-1) * ood_data['probs'], -1)
    ood_norm = ood_data['conf']
    labels = np.concatenate((np.zeros(len(id_nak)), np.ones(len(ood_nak))))
    print(fname)
    print("NAK")
    print(roc_auc_score(labels, np.concatenate((id_nak, ood_nak))))
    print("L2")
    print(roc_auc_score(1 - labels, np.concatenate((id_norm, ood_norm))))


