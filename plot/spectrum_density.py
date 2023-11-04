import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from colour import Color
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='name of spectrum')
parser.add_argument('-o', '--out', help='name of out file')
args = parser.parse_args()

spec_dir = '/h/nng/projects/OpenOOD/results/ekfac'

spec = torch.load(os.path.join(spec_dir, args.name + '_spec.cpt'), map_location='cpu')

blue = Color('blue')
colors = list(blue.range_to(Color('green'), len(spec)))

for s, c in zip(spec, colors):
    sns.kdeplot(np.log(s.numpy()), color=str(c), bw_adjust=0.5, fill=True, alpha=0.2)

plt.savefig(os.path.join(spec_dir, args.out))
