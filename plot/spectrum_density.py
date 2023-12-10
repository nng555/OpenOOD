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

fig1, ax1 = plt.subplots()

for i, (s, c) in enumerate(zip(spec, colors)):
    fig2, ax2 = plt.subplots()
    dens = np.log10(s.numpy())
    dens[np.isnan(dens)] = -80
    sns.kdeplot(dens, color=str(c), bw_adjust=0.5, fill=True, alpha=0.8, ax=ax2)
    sns.kdeplot(dens, color=str(c), bw_adjust=0.5, fill=True, alpha=0.1, ax=ax1)
    ax2.axvline(x = np.log10(np.median(s)))
    fig2.savefig(os.path.join(spec_dir, args.out + f'_{i}.png'))
    plt.close(fig2)
fig1.savefig(os.path.join(spec_dir, args.out + '.png'))
