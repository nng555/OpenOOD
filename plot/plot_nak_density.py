import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns

slurm_dir = '/h/nng/slurm'
res_dir = 'results/cifar10_resnet18_32x32_test_ood_ood_nak_default/s0/ood/scores'

parser = argparse.ArgumentParser()
parser.add_argument('-n1', '--name1', help='name of first dir')
parser.add_argument('-d', '--dataset', help='name of dataset to plot')
args = parser.parse_args()

d1 = np.load(os.path.join(slurm_dir, args.name1, res_dir, args.dataset + '.npz'), allow_pickle=True)
sns.kdeplot(np.log(-d1['conf']), bw_adjust=0.5, fill=True, alpha=0.2)
plt.savefig('out.png')

import ipdb; ipdb.set_trace()
