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
id_data = np.load(os.path.join(ddir, args.dataset + '.npz'), allow_pickle=True)['extra'].item()
id_scores = id_data['self_nak']
id_logits = id_data['logits']
for fname in os.listdir(ddir):
    if fname.split('.npz')[0] == args.dataset:
        continue
    print(fname)
    ood_data = np.load(os.path.join(ddir, fname), allow_pickle=True)['extra'].item()
    ood_scores = ood_data['self_nak']
    ood_logits = ood_data['logits']
    labels = np.concatenate((np.zeros(len(id_scores)), np.ones(len(ood_scores))))
    print(roc_auc_score(labels, np.concatenate((id_scores[..., :-1].sum(-1).min(-1), ood_scores[..., :-1].sum(-1).min(-1)))))

import ipdb; ipdb.set_trace()
